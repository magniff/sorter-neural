#![allow(clippy::single_range_in_vec_init)]
#![recursion_limit = "256"]

use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::AdamConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder};
use burn::tensor::{Tensor, TensorData, backend::AutodiffBackend};

// ─── Knobs ──────────────────────────────────────────────────────────
const INPUT_SIZE: usize = 16;
const OUTPUT_SIZE: usize = INPUT_SIZE;
const SCORE_SIZE: usize = INPUT_SIZE * OUTPUT_SIZE; // 256 — the flattened score matrix

// The score-producing MLP can be much smaller than level 0.
// It only needs to learn relative ordering, not output values.
const HIDDEN_WIDTH: usize = 128;
const NUM_HIDDEN: usize = 4;

const LEARNING_RATE: f64 = 1e-3;
const BATCH_SIZE: usize = 1024;
const BATCHES_PER_EPOCH: usize = 100;
const VAL_BATCHES: usize = 20;
const SEED: u64 = 42;

// Sinkhorn parameters
const SINKHORN_ITERS: usize = 20;

// Fixed moderate tau for training — keeps gradients healthy.
// No annealing: the network learns good scores at this temperature,
// and at eval we extract a hard permutation from those same scores.
const TAU_TRAIN: f32 = 0.1;

// Logging / checkpointing
const LOG_INTERVAL: usize = 10;
const SAMPLE_INTERVAL: usize = 200;
const CHECKPOINT_INTERVAL: usize = 200;
const NUM_SAMPLES: usize = 8;
const SET_TOLERANCE: f32 = 0.005;

const CHECKPOINT_DIR: &str = "checkpoints";
const CHECKPOINT_BASE: &str = "checkpoints/sort_sinkhorn";

// ─── Model ──────────────────────────────────────────────────────────

#[derive(Module, Debug)]
struct SinkhornSortNet<B: Backend> {
    input_layer: Linear<B>,
    hidden_layers: Vec<Linear<B>>,
    output_layer: Linear<B>,
    relu: Relu,
}

impl<B: Backend> SinkhornSortNet<B> {
    fn new(device: &B::Device) -> Self {
        let input_layer = LinearConfig::new(INPUT_SIZE, HIDDEN_WIDTH).init(device);
        let hidden_layers = (0..NUM_HIDDEN)
            .map(|_| LinearConfig::new(HIDDEN_WIDTH, HIDDEN_WIDTH).init(device))
            .collect();
        let output_layer = LinearConfig::new(HIDDEN_WIDTH, SCORE_SIZE).init(device);
        Self {
            input_layer,
            hidden_layers,
            output_layer,
            relu: Relu::new(),
        }
    }

    /// Produce raw score matrix [B, 16, 16] from input.
    fn scores(&self, x: &Tensor<B, 2>) -> Tensor<B, 3> {
        let bs = x.dims()[0];
        let mut h = self.relu.forward(self.input_layer.forward(x.clone()));
        for layer in &self.hidden_layers {
            h = self.relu.forward(layer.forward(h));
        }
        let raw = self.output_layer.forward(h); // [B, 256]
        raw.reshape([bs as i32, INPUT_SIZE as i32, OUTPUT_SIZE as i32])
    }

    /// Training forward: returns both soft output and the P matrix.
    /// We need P for the ordering loss.
    fn forward_soft_with_p(&self, x: Tensor<B, 2>, tau: f32) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let bs = x.dims()[0];
        let scores = self.scores(&x);

        let scaled = (scores / tau).clamp(-50.0, 50.0);
        let s = scaled.exp();
        let p = sinkhorn::<B>(s, SINKHORN_ITERS);

        let x_col = x.reshape([bs as i32, INPUT_SIZE as i32, 1]);
        let out = p.clone().matmul(x_col);
        let out = out.reshape([bs as i32, OUTPUT_SIZE as i32]);
        (out, p)
    }

    /// Eval forward: hard permutation via argmax on Sinkhorn output.
    /// Each output position picks exactly one input element.
    /// NOT differentiable — only for evaluation.
    fn forward_hard(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let bs = x.dims()[0];
        let scores = self.scores(&x);

        // Run Sinkhorn at moderate tau to get a good doubly-stochastic matrix,
        // then extract hard assignments via argmax per row.
        let scaled = (scores / TAU_TRAIN).clamp(-50.0, 50.0);
        let s = scaled.exp();
        let p_soft = sinkhorn::<B>(s, SINKHORN_ITERS);

        // Extract hard permutation on CPU
        let p_data: Vec<f32> = p_soft.to_data().to_vec().unwrap();
        let x_data: Vec<f32> = x.to_data().to_vec().unwrap();
        let mut out_flat = vec![0.0f32; bs * OUTPUT_SIZE];

        for b in 0..bs {
            let mut used = [false; INPUT_SIZE]; // track which inputs are taken

            // Greedy assignment: process rows in order of confidence
            // (row with highest max value gets first pick)
            let mut row_order: Vec<usize> = (0..OUTPUT_SIZE).collect();
            row_order.sort_by(|&r1, &r2| {
                let max1 = (0..INPUT_SIZE)
                    .map(|c| p_data[b * SCORE_SIZE + r1 * INPUT_SIZE + c])
                    .fold(0.0f32, f32::max);
                let max2 = (0..INPUT_SIZE)
                    .map(|c| p_data[b * SCORE_SIZE + r2 * INPUT_SIZE + c])
                    .fold(0.0f32, f32::max);
                max2.partial_cmp(&max1).unwrap_or(std::cmp::Ordering::Equal)
            });

            for &row in &row_order {
                // Find best unused column for this row
                let mut best_col = 0;
                let mut best_val = f32::NEG_INFINITY;
                for col in 0..INPUT_SIZE {
                    if !used[col] {
                        let v = p_data[b * SCORE_SIZE + row * INPUT_SIZE + col];
                        if v > best_val {
                            best_val = v;
                            best_col = col;
                        }
                    }
                }
                used[best_col] = true;
                out_flat[b * OUTPUT_SIZE + row] = x_data[b * INPUT_SIZE + best_col];
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(out_flat, [bs, OUTPUT_SIZE]), &x.device())
    }
}

/// Sinkhorn iteration: alternate row and column normalization.
/// Converges to a doubly stochastic matrix (every row and column sums to 1).
///
/// This is the entire "structural prior" — it constrains the network
/// output to live on the Birkhoff polytope (convex hull of permutation
/// matrices). As temperature → 0, the vertices of this polytope
/// (actual permutations) are approached.
fn sinkhorn<B: Backend>(mut s: Tensor<B, 3>, iters: usize) -> Tensor<B, 3> {
    for _ in 0..iters {
        // Row normalization: divide each element by its row sum
        let row_sums = s.clone().sum_dim(2); // [B, 16, 1] — keepdim
        s = s / (row_sums + 1e-6);

        // Column normalization: divide each element by its column sum
        let col_sums = s.clone().sum_dim(1); // [B, 1, 16] — keepdim
        s = s / (col_sums + 1e-6);
    }
    s
}

// ─── Checkpointing ─────────────────────────────────────────────────
type JsonRecorder = PrettyJsonFileRecorder<FullPrecisionSettings>;

fn save_checkpoint<B: Backend>(model: &SinkhornSortNet<B>, epoch: usize) {
    std::fs::create_dir_all(CHECKPOINT_DIR).ok();
    let recorder = JsonRecorder::new();

    model
        .clone()
        .save_file(CHECKPOINT_BASE, &recorder)
        .expect("failed to save checkpoint");

    let numbered = format!("{CHECKPOINT_DIR}/sort_sinkhorn_epoch_{epoch}");
    model
        .clone()
        .save_file(&numbered, &recorder)
        .expect("failed to save numbered checkpoint");

    println!("  [checkpoint] saved epoch {epoch} → {CHECKPOINT_BASE}.json + {numbered}.json");
}

fn try_load_checkpoint<B: Backend>(
    model: SinkhornSortNet<B>,
    device: &B::Device,
) -> (SinkhornSortNet<B>, bool) {
    let recorder = JsonRecorder::new();
    match model.load_file(CHECKPOINT_BASE, &recorder, device) {
        Ok(loaded) => {
            println!("  [checkpoint] resumed from {CHECKPOINT_BASE}.json");
            (loaded, true)
        }
        Err(_) => {
            println!("  [checkpoint] no checkpoint found, starting fresh");
            (SinkhornSortNet::new(device), false)
        }
    }
}

// ─── Data generation ────────────────────────────────────────────────
fn generate_batch<B: Backend>(
    batch_size: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut inputs_flat = Vec::with_capacity(batch_size * INPUT_SIZE);
    let mut targets_flat = Vec::with_capacity(batch_size * OUTPUT_SIZE);

    for _ in 0..batch_size {
        let mut vals: Vec<f32> = (0..INPUT_SIZE).map(|_| rand::random::<f32>()).collect();
        inputs_flat.extend_from_slice(&vals);
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        targets_flat.extend_from_slice(&vals);
    }

    let inputs = Tensor::<B, 2>::from_data(
        TensorData::new(inputs_flat, [batch_size, INPUT_SIZE]),
        device,
    );
    let targets = Tensor::<B, 2>::from_data(
        TensorData::new(targets_flat, [batch_size, OUTPUT_SIZE]),
        device,
    );
    (inputs, targets)
}

// ─── Loss ───────────────────────────────────────────────────────────
fn mse_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - target;
    let sq = diff.clone() * diff;
    sq.mean().reshape([1])
}

/// Pairwise ordering loss on the soft permutation matrix.
///
/// From P (doubly stochastic, [B, N, N]), compute the "expected output
/// position" for each input element j:
///
///   pos_j = sum_i  i * P[i][j]     (weighted average of row indices)
///
/// Then for every pair (j, k) where input[j] < input[k], we want
/// pos_j < pos_k. Penalize violations with a soft hinge:
///
///   loss += max(0, pos_j - pos_k + margin)
///
/// This directly teaches the score matrix to produce correct orderings,
/// giving gradient signal even for close-valued elements that MSE ignores.
fn ordering_loss<B: Backend>(
    p_soft: Tensor<B, 3>,  // [B, N, N] doubly stochastic
    inputs: &Tensor<B, 2>, // [B, N] raw inputs
) -> Tensor<B, 1> {
    let bs = inputs.dims()[0];
    let n = INPUT_SIZE;
    let margin = 0.1f32;

    // Position indices [0, 1, 2, ..., 15] as a column vector [N, 1]
    let pos_indices: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let pos_vec = Tensor::<B, 2>::from_data(TensorData::new(pos_indices, [n, 1]), &inputs.device()); // [N, 1]

    // Expected position for each input element:
    // P is [B, N_out, N_in], so P^T is [B, N_in, N_out]
    // expected_pos_j = sum_i P[i][j] * i = (P^T · pos_indices)[j]
    let p_t = p_soft.swap_dims(1, 2); // [B, N_in, N_out]
    let expected_pos = p_t.matmul(pos_vec.unsqueeze::<3>()); // [B, N_in, 1]
    let expected_pos = expected_pos.squeeze::<2>(); // [B, N_in]

    // Get input values to determine ground-truth ordering
    let inp_data: Vec<f32> = inputs.to_data().to_vec().unwrap();

    // For each sample, build pairwise hinge losses for a random subset
    // of pairs (doing all N*(N-1)/2 = 120 pairs is fine for N=16)
    let mut hinge_total = Vec::new();

    for b in 0..bs {
        let row = &inp_data[b * n..(b + 1) * n];
        for j in 0..n {
            for k in (j + 1)..n {
                // We want: if input[j] < input[k], then pos[j] < pos[k]
                // Penalize: pos[j] - pos[k] + margin > 0 when j should come first
                // And vice versa
                if row[j] < row[k] {
                    // j should have smaller position than k
                    hinge_total.push((b, j, k)); // want pos_j < pos_k
                } else {
                    hinge_total.push((b, k, j)); // want pos_k < pos_j
                }
            }
        }
    }

    // Gather expected positions for all pairs and compute hinge
    // We'll do this via tensor indexing: build tensors of pos[earlier] and pos[later]
    let num_pairs = hinge_total.len();
    let earlier_idx: Vec<f32> = hinge_total
        .iter()
        .map(|&(b, j, _)| {
            // index into flat expected_pos
            (b * n + j) as f32
        })
        .collect();
    let later_idx: Vec<f32> = hinge_total
        .iter()
        .map(|&(b, _, k)| (b * n + k) as f32)
        .collect();

    // Flatten expected_pos to [B*N] for gathering
    let ep_flat = expected_pos.reshape([bs as i32 * n as i32]);

    let earlier_indices = Tensor::<B, 1, burn::tensor::Int>::from_data(
        TensorData::new(
            earlier_idx.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            [num_pairs],
        ),
        &inputs.device(),
    );
    let later_indices = Tensor::<B, 1, burn::tensor::Int>::from_data(
        TensorData::new(
            later_idx.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            [num_pairs],
        ),
        &inputs.device(),
    );

    let pos_earlier = ep_flat.clone().select(0, earlier_indices);
    let pos_later = ep_flat.select(0, later_indices);

    // Hinge: max(0, pos_earlier - pos_later + margin)
    let violation = (pos_earlier - pos_later) + margin;
    let hinge = violation.clamp_min(0.0);
    hinge.mean().reshape([1])
}

const MSE_WEIGHT: f32 = 1.0;
const ORD_WEIGHT: f32 = 1.0;

// ─── Metrics ────────────────────────────────────────────────────────
fn element_accuracy<B: Backend>(pred: &Tensor<B, 2>, target: &Tensor<B, 2>, tol: f32) -> f32 {
    let diff = pred.clone() - target.clone();
    let within = diff.abs().lower_elem(tol).float().mean();
    let val: f32 = within.into_scalar().elem();
    val * 100.0
}

fn fraction_correctly_sorted<B: Backend>(
    pred: &Tensor<B, 2>,
    input: &Tensor<B, 2>,
) -> (f32, f32, f32) {
    let pred_flat: Vec<f32> = pred.to_data().to_vec().unwrap();
    let input_flat: Vec<f32> = input.to_data().to_vec().unwrap();
    let bs = pred_flat.len() / OUTPUT_SIZE;

    let mut n_monotonic = 0usize;
    let mut n_set_match = 0usize;
    let mut n_both = 0usize;

    for b in 0..bs {
        let s = b * OUTPUT_SIZE;
        let e = s + OUTPUT_SIZE;
        let row = &pred_flat[s..e];
        let inp = &input_flat[s..e];

        let monotonic = row.windows(2).all(|w| w[0] <= w[1]);

        let mut p_sorted: Vec<f32> = row.to_vec();
        let mut i_sorted: Vec<f32> = inp.to_vec();
        p_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        i_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let set_ok = p_sorted
            .iter()
            .zip(i_sorted.iter())
            .all(|(p, t)| (p - t).abs() < SET_TOLERANCE);

        if monotonic {
            n_monotonic += 1;
        }
        if set_ok {
            n_set_match += 1;
        }
        if monotonic && set_ok {
            n_both += 1;
        }
    }
    let bs_f = bs as f32;
    (
        n_monotonic as f32 / bs_f * 100.0,
        n_set_match as f32 / bs_f * 100.0,
        n_both as f32 / bs_f * 100.0,
    )
}

fn set_match_error<B: Backend>(pred: &Tensor<B, 2>, input: &Tensor<B, 2>) -> f32 {
    let pred_flat: Vec<f32> = pred.to_data().to_vec().unwrap();
    let input_flat: Vec<f32> = input.to_data().to_vec().unwrap();
    let bs = pred_flat.len() / OUTPUT_SIZE;

    let mut total = 0.0f32;
    for b in 0..bs {
        let s = b * OUTPUT_SIZE;
        let e = s + OUTPUT_SIZE;
        let mut p: Vec<f32> = pred_flat[s..e].to_vec();
        let mut t: Vec<f32> = input_flat[s..e].to_vec();
        p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mae: f32 = p
            .iter()
            .zip(t.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / OUTPUT_SIZE as f32;
        total += mae;
    }
    total / bs as f32
}

// ─── Temperature schedule ───────────────────────────────────────────

// (no temperature schedule — fixed tau for training)

// ─── Training ───────────────────────────────────────────────────────

fn train<B: AutodiffBackend>(device: B::Device) {
    B::seed(&device, SEED);

    let model_fresh = SinkhornSortNet::<B>::new(&device);
    let resume = std::env::args().any(|a| a == "--resume");

    let (mut model, resumed) = if resume {
        try_load_checkpoint(model_fresh, &device)
    } else {
        println!("  [checkpoint] starting fresh (pass --resume to load checkpoint)");
        (model_fresh, false)
    };

    let mut optim = AdamConfig::new().init();

    let param_count: usize = model.num_params();
    println!("=== Level 1: Sinkhorn Permutation Sorting Network ===");
    println!("  Input/Output size : {INPUT_SIZE}");
    println!("  Score matrix      : {INPUT_SIZE}x{OUTPUT_SIZE} = {SCORE_SIZE} outputs");
    println!("  Hidden width      : {HIDDEN_WIDTH}");
    println!("  Hidden layers     : {NUM_HIDDEN}");
    println!("  Parameters        : {param_count}");
    println!("  Batch size        : {BATCH_SIZE}");
    println!("  Learning rate     : {LEARNING_RATE}");
    println!("  Sinkhorn iters    : {SINKHORN_ITERS}");
    println!("  Tau (train)       : {TAU_TRAIN} (fixed)");
    println!("  Loss weights      : mse={MSE_WEIGHT}, ordering={ORD_WEIGHT}");
    println!("  Eval              : hard argmax permutation");
    println!("  Set tolerance     : {SET_TOLERANCE}");
    println!("  Checkpoint every  : {CHECKPOINT_INTERVAL} epochs");
    println!("  Resumed           : {resumed}");
    println!("  Running forever   : Ctrl+C to stop");
    println!();

    let mut epoch = 0usize;
    loop {
        epoch += 1;
        let mut epoch_loss = 0.0f32;
        let mut epoch_mse = 0.0f32;
        let mut epoch_ord = 0.0f32;

        for _ in 0..BATCHES_PER_EPOCH {
            let (inputs, targets) = generate_batch::<B>(BATCH_SIZE, &device);
            let (pred, p_soft) = model.forward_soft_with_p(inputs.clone(), TAU_TRAIN);

            let l_mse = mse_loss(pred, targets);
            let l_ord = ordering_loss(p_soft, &inputs);

            let mse_val: f32 = l_mse.clone().into_scalar().elem();
            let ord_val: f32 = l_ord.clone().into_scalar().elem();

            let loss = l_mse * MSE_WEIGHT + l_ord * ORD_WEIGHT;
            epoch_loss += loss.clone().into_scalar().elem::<f32>();
            epoch_mse += mse_val;
            epoch_ord += ord_val;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);
        }
        epoch_loss /= BATCHES_PER_EPOCH as f32;
        epoch_mse /= BATCHES_PER_EPOCH as f32;
        epoch_ord /= BATCHES_PER_EPOCH as f32;

        // ── Eval: use hard argmax permutation ──
        if epoch % LOG_INTERVAL == 0 || epoch == 1 {
            let model_valid = model.valid();
            let mut val_loss = 0.0f32;
            let mut val_acc = 0.0f32;
            let mut val_mono = 0.0f32;
            let mut val_set_pct = 0.0f32;
            let mut val_correct = 0.0f32;
            let mut val_set_err = 0.0f32;

            for _ in 0..VAL_BATCHES {
                let (inputs, targets) = generate_batch::<B::InnerBackend>(BATCH_SIZE, &device);
                let pred = model_valid.forward_hard(inputs.clone());

                let lv: f32 = mse_loss(pred.clone(), targets.clone()).into_scalar().elem();
                val_loss += lv;
                val_acc += element_accuracy(&pred, &targets, 0.05);
                let (mono, set_pct, both) = fraction_correctly_sorted(&pred, &inputs);
                val_mono += mono;
                val_set_pct += set_pct;
                val_correct += both;
                val_set_err += set_match_error(&pred, &inputs);
            }
            val_loss /= VAL_BATCHES as f32;
            val_acc /= VAL_BATCHES as f32;
            val_mono /= VAL_BATCHES as f32;
            val_set_pct /= VAL_BATCHES as f32;
            val_correct /= VAL_BATCHES as f32;
            val_set_err /= VAL_BATCHES as f32;

            println!(
                "Epoch {epoch:>6} | loss: {epoch_loss:.6} (mse={epoch_mse:.6} \
                 ord={epoch_ord:.4}) | val_mse: {val_loss:.6} | \
                 acc(<.05): {val_acc:.1}% | mono: {val_mono:.1}% | \
                 set_ok: {val_set_pct:.1}% | CORRECT: {val_correct:.1}% | \
                 set_mae: {val_set_err:.5}"
            );
        }

        // ── Checkpoint ──
        if epoch % CHECKPOINT_INTERVAL == 0 {
            save_checkpoint(&model, epoch);
        }

        // ── Sample test ──
        if epoch % SAMPLE_INTERVAL == 0 {
            print_samples::<B>(&model, &device, epoch);
        }
    }
}

fn print_samples<B: AutodiffBackend>(
    model: &SinkhornSortNet<B>,
    device: &B::Device,
    _epoch: usize,
) {
    println!("\n────────────────────── Sample Test (hard permutation) ──────────────────────");
    let model_valid = model.valid();
    let (inputs, _targets) = generate_batch::<B::InnerBackend>(NUM_SAMPLES, device);
    let pred = model_valid.forward_hard(inputs.clone());

    let inp_data: Vec<f32> = inputs.to_data().to_vec().unwrap();
    let prd_data: Vec<f32> = pred.to_data().to_vec().unwrap();

    let mut n_correct = 0usize;

    for i in 0..NUM_SAMPLES {
        let s = i * INPUT_SIZE;
        let e = s + INPUT_SIZE;

        let row_pred = &prd_data[s..e];
        let row_input = &inp_data[s..e];

        let mut pred_sorted: Vec<f32> = row_pred.to_vec();
        pred_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut input_sorted: Vec<f32> = row_input.to_vec();
        input_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let monotonic = row_pred.windows(2).all(|w| w[0] <= w[1]);
        let set_ok = pred_sorted
            .iter()
            .zip(input_sorted.iter())
            .all(|(p, t)| (p - t).abs() < SET_TOLERANCE);
        let correct = monotonic && set_ok;
        if correct {
            n_correct += 1;
        }

        let set_mae: f32 = pred_sorted
            .iter()
            .zip(input_sorted.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / OUTPUT_SIZE as f32;

        let max_elem_err: f32 = pred_sorted
            .iter()
            .zip(input_sorted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let status = match (monotonic, set_ok) {
            (true, true) => "CORRECT",
            (true, false) => "wrong values",
            (false, true) => "wrong order",
            (false, false) => "wrong both",
        };

        println!("\n  Sample {i}: [{status}]  set_mae={set_mae:.5}  max_err={max_elem_err:.5}");
        println!("    input:      {:.4?}", row_input);
        println!("    target:     {:.4?}", &input_sorted);
        println!("    pred:       {:.4?}", row_pred);
        println!("    pred(sort): {:.4?}", &pred_sorted);
    }
    println!(
        "\n  >>> {n_correct}/{NUM_SAMPLES} correctly sorted (monotonic + same set within {SET_TOLERANCE})"
    );
    println!("──────────────────────────────────────────────────────────\n");
}

fn main() {
    let device = WgpuDevice::default();
    println!("Using WGPU backend (will use GPU if available)\n");
    train::<Autodiff<Wgpu>>(device);
}
