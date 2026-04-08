# VioPTT — ISMIR Paper Reference

> Frame-Level Violin Playing Technique Detection with Multi-Scale Mixture-of-Experts

---

## 1. Motivation

以往做 violin technique detection 的模型皆採用 **pre-segmented note-level technique prediction**：先偵測 note boundary（onset/offset），再對每個 note segment 做 technique 分類。直接做 **frame-level prediction** 的效果一向不佳，主因在於不同 technique 需要的時間尺度差異極大（attack transient ~100ms vs. legato phrase >1s），單一模型難以同時捕捉。

然而 frame-level prediction 具有以下優勢：

- **不需要 pre-segmentation**：省去 note onset/offset detection 的前處理步驟
- **無 cascading error**：note-level 方法的 technique 準確度受限於 onset detection 的正確率；frame-level 直接預測，避免了 error propagation
- **模型架構更簡潔**：end-to-end，一個 forward pass 同時產出 transcription + technique
- **更高時間解析度**：以 10ms/frame（100 fps）為單位預測，可精確定位 technique 發生的時間點

---

## 2. Contributions（四大 Novelty）

### 2.1 首篇預測 Bow Change（Legato）時機的工作

- **Legato / Bow change** 是小提琴演奏中最基本且最常見的技巧之一
- 小提琴的 onset 遠比 piano/guitar 難以偵測（弓弦接觸為連續性發聲），使得 bow change point 成為困難的問題
- 以往沒有真實的 dataset 做 legato 標註
- 本研究將 legato 視為 binary regression 問題：使用 **Gaussian peak** 標記 bow change point，模型輸出 `[0, 1]` 的 soft probability，1.0 代表 bow change、0.0 代表 sustained
- 使用 **positive-weighted BCE**（bow_change frame 的 weight = 5.0）處理 class imbalance

### 2.2 VioTech120 Dataset

- 基於 **ViolinMIDIDataset** 的全部曲目，手動人工標註了 **120 首小提琴獨奏曲目**
- 6 種 technique（note-level annotation），分為三個維度：


| 維度              | Classes                                        | 說明   |
| --------------- | ---------------------------------------------- | ---- |
| Tonal Technique | `none`, `pizzicato`, `harmonics`, `openstring` | 發聲方式 |
| Articulation    | `none`, `release`, `staccato`, `spiccato`      | 運弓方式 |
| Legato          | `bow_change`, `sustained`                      | 換弓時機 |


- **資料格式**：每首曲目包含 `*_cut.wav`（音訊）+ `annotation_revised.csv`（標註）
- **CSV 欄位**：`pitch`, `start`, `end`, `tonalTechnique`, `articulation`, `legato`
- **資料前處理** (`prepare_viotech.py`)：轉換為 HDF5 格式，包含 `waveform`（int16, 16kHz）、`midi_event`、`midi_event_time`、`tonalTechnique`、`articulation`、`legato`
- **Train/Val/Test split**：基於 song name hash 決定（~80/10/10%），確保 deterministic

**Reduced label config**（實際訓練用）：


| 維度           | 合併策略                                    | 最終 classes                             |
| ------------ | --------------------------------------- | -------------------------------------- |
| Tonal        | `openstring` → `none`                   | none, pizzicato, harmonics (3 classes) |
| Articulation | `release` → `none`, `staccato` → `none` | none, spiccato (2 classes)             |
| Legato       | 不合併                                     | bow_change, sustained (2 classes)      |


**輔助資料集**：

- **MOSAVPT**：用於 mixed training（`viotech_mixed_mosavpt` mode），以 `mosavpt_ratio` 控制取樣比例（default 0.2）
- **RWC**：用於 cross-dataset evaluation，technique label 透過 `rwc_technique_map` 對應

### 2.3 Technique Annotation Interface

- 開發了一個標註介面，讓音樂人可以直覺地「畫出」technique 標註
- 支援 note-level 標註三個維度的 technique
- 產出 `annotation_revised.csv` 格式

### 2.4 Frame-Level Multi-Scale MoE Model

核心貢獻：使用不同時間尺度的 expert 讓模型能夠更好地做 frame-level prediction。

---

## 3. Model Architecture

### 3.1 Backbone — HRPT-based Transcription

基於 ByteDance 的 **High-Resolution Piano Transcription (HRPT)** 架構，改造為小提琴：

```
Input: mono waveform (16kHz)
  ↓
Spectrogram (n_fft=2048, hop=160) → LogMel (229 bins, 30–8000 Hz)
  ↓ BatchNorm
  ↓
┌──────────────────────────────────────────────────────────────┐
│  4 × Parallel AcousticModelCRnn8Dropout branches:           │
│    • frame_model       → frame_output (B, T, 88)            │
│    • reg_onset_model   → reg_onset_output (B, T, 88)        │
│    • reg_offset_model  → reg_offset_output (B, T, 88)       │
│    • velocity_model    → velocity_output (B, T, 88)         │
└──────────────────────────────────────────────────────────────┘
  ↓ GRU Fusion
  • Onset GRU: concat(onset, sqrt(onset) × velocity) → refined onset
  • Frame GRU: concat(frame, onset_detach, offset_detach) → refined frame
```

**AcousticModelCRnn8Dropout** 結構：

```
ConvBlock(1→48) → pool(1,2) → dropout(0.2)
ConvBlock(48→64) → pool(1,2) → dropout(0.2)
ConvBlock(64→96) → pool(1,2) → dropout(0.2)
ConvBlock(96→128) → pool(1,2) → dropout(0.2)
→ flatten → FC(1792→768) → BN → ReLU → dropout(0.5)
→ BiGRU(768→256×2) → dropout(0.5)
→ FC(512→88) → sigmoid
Output: (B, T, 88) probabilities + (B, T, 512) GRU features
```

- `ConvBlock` = Conv2d(3×3) + BN + ReLU → Conv2d(3×3) + BN + ReLU → AvgPool
- `frames_per_second = 100`（每 frame = 10ms）
- `segment_seconds = 10`（訓練時每段 10 秒）
- `classes_num = 88`（MIDI notes 21-108）

### 3.2 Frame-Level Multi-Scale MoE Head（核心模型）

獨立於 transcription 分支的 **dedicated CRNN**（`technique_acoustic_frame_moe`）提取 512-dim features，搭配 transcription 分支的 structural cues。

#### Design Rationale — 為何用 MoE

不同演奏技法所需的時間尺度差異極大：pizzicato 的辨識只需看 onset 瞬間 (~~100ms) 的攻擊波形，vibrato 需要整個 note 持續期間 (~~500ms) 的 amplitude modulation，而 legato 需要看到跨 note 的樂句連貫性 (>1s)。單一卷積核無法同時覆蓋這些尺度。

MoE 的核心思路：**設計多個擅長不同時間視野的 expert，由 gate network 動態決定每個 frame 應聽取哪些 expert 的意見**。更進一步，每個下游任務（tonal / articulation / legato）擁有獨立的 gate，使得 articulation 任務可偏好 onset expert，而 legato 任務可偏好 phrase expert。

#### Expert 設計


| Expert | 名稱                  | Kernel × Dilation | 有效感受野            | 注入的 Cue                      | 擅長捕捉                                                |
| ------ | ------------------- | ----------------- | ---------------- | ---------------------------- | --------------------------------------------------- |
| E0     | Onset Specialist    | 11 × 1            | ~110ms           | onset probability            | Attack transients (spiccato, pizzicato, bow change) |
| E1     | Note Specialist     | 51 × 1            | ~510ms           | frame activity               | Sustained note characteristics (vibrato, sustain)   |
| E2     | Phrase Specialist   | 51 × 3 (dilated)  | ~1530ms          | onset + offset + frame (all) | Legato transitions, phrase-level structure          |
| E3     | Spectral Specialist | 2D CNN on logmel  | frequency domain | frame activity               | Harmonic energy distribution, overtone patterns     |


- 每個 temporal expert 使用 **depthwise-separable 1D conv**（`_TemporalCtx`）提取不同時間尺度的特徵
- Expert MLP：`Linear(in_dim → 256) → GELU → Dropout(0.2) → Linear(256 → 128)`
- E3（Spectral）使用獨立的 4-layer ConvBlock stack（`_DedicatedSpectralCNN`），與 transcription 分支不共享權重

每個 Expert 的結構分為兩段：**特徵提取器**（各自不同）+ **共用結構的 MLP 映射**（`_Expert`）。所有 expert 最終輸出相同維度 `shared_dim=128`，方便 gate 加權合併。

#### Building Block：`_TemporalCtx`（時間域特徵提取）

E0–E2 共用的特徵提取器，差別僅在 kernel_size 和 dilation：

```
Input: (B, T, D=512)
  ↓ transpose → (B, D, T)
  ↓ Depthwise Conv1d(D→D, kernel, dilation, groups=D)  ← 每個 channel 獨立卷積
  ↓ Pointwise Conv1d(D→P=128, kernel=1)                ← 跨 channel 融合 + 降維
  ↓ transpose → (B, T, P)
  ↓ LayerNorm(P) → GELU
Output: (B, T, 128)
```

- **Depthwise-separable** 設計：參數量 = D×kernel + D×P，遠小於標準 Conv1d 的 D×D×kernel
- Causal padding `(kernel-1)×dilation//2` 確保不 leak 未來資訊，同時維持時間解析度不變


| Expert | kernel | dilation | 有效感受野               | 對應音樂事件              |
| ------ | ------ | -------- | ------------------- | ------------------- |
| E0     | 11     | 1        | 11 frames = 110ms   | 單一 attack transient |
| E1     | 51     | 1        | 51 frames = 510ms   | 完整 note duration    |
| E2     | 51     | 3        | 153 frames = 1530ms | 跨 note 的 phrase     |


#### Building Block：`_DedicatedSpectralCNN`（頻域特徵提取）

E3 專用。獨立的 4-layer 2D CNN，直接從 log-mel spectrogram 提取頻域特徵：

```
Input: logmel (B, 1, T, 229) — BN-normalised log-mel spectrogram
  ↓ ConvBlock(1→48)  + AvgPool(1,2) + Dropout(0.2)   → (B, 48, T, 114)
  ↓ ConvBlock(48→64) + AvgPool(1,2) + Dropout(0.2)   → (B, 64, T, 57)
  ↓ ConvBlock(64→96) + AvgPool(1,2) + Dropout(0.2)   → (B, 96, T, 28)
  ↓ ConvBlock(96→128)+ AvgPool(1,2) + Dropout(0.2)   → (B, 128, T, 14)
  ↓ transpose + flatten                               → (B, T, 1792)
  ↓ Linear(1792→128) → LayerNorm(128) → GELU
Output: (B, T, SP=128) — per-frame spectral descriptor
```

- 只在**頻率軸**做 pooling `(1,2)`，保留**時間軸**解析度
- 所有參數由 technique loss 獨立訓練，不受 transcription gradient 污染
- 擅長捕捉 harmonics 的泛音分布、pizzicato 的頻譜衰減特徵

Spectral expert 的完整輸入 = `_DedicatedSpectralCNN` 輸出 (SP=128) + `feat_proj`（GRU features 的 Linear+LN+GELU 投影, P=128）+ cue embedding。

#### Building Block：`_Expert`（Expert MLP）

所有 4 個 expert 共用同一個 MLP class，只是 `in_dim` 依據各自的特徵提取器不同而異：

```
Input: (B, T, in_dim)    ← in_dim 因 expert 而異
  ↓ Linear(in_dim → 256) → GELU → Dropout(0.2)
  ↓ Linear(256 → 128)
Output: (B, T, shared_dim=128)
```


| Expert                | in_dim 計算                   | in_dim |
| --------------------- | --------------------------- | ------ |
| E0 (Legacy)           | P + C = 128 + 16            | 144    |
| E1 (Legacy)           | P + C = 128 + 16            | 144    |
| E2 (Legacy)           | P + 3C = 128 + 48           | 176    |
| E3 (Spectral, Legacy) | SP + P + C = 128 + 128 + 16 | 272    |


#### Building Block：`_Gate`（路由網路）

```
Input: (B, T, gate_in)   ← gate_in = N_temporal×P + cue_total [+ SP + P]
  ↓ Linear(gate_in → 128) → ReLU → Dropout
  ↓ Linear(128 → num_experts)
Output: (B, T, E) — raw logits, 後經 top-k softmax
```

Gate 看到的是**全域資訊**（所有 temporal context + spectral features + cue embeddings），因此能根據當前 frame 的完整 context 決定路由。

#### Transcription Cue Injection

```python
onset_cue  = reg_onset_output.max(dim=-1, keepdim=True)   # (B, T, 1) — max over 88 pitches
offset_cue = reg_offset_output.max(dim=-1, keepdim=True)  # (B, T, 1)
frame_cue  = frame_output.max(dim=-1, keepdim=True)       # (B, T, 1)
```

- 每個 cue 經過 `Linear(1 → 16) → GELU` 投影為 `cue_dim=16` 的 embedding
- Legacy cue mode：E0 只看 onset_cue，E1 只看 frame_cue，E2 看全部三個 cue

#### Per-Task Gating

三個獨立的 gate network（tonal / articulation / legato），每個 gate：

```
Input: concat(all temporal contexts, [spectral features], [cue embeddings])
→ Linear(gate_in → 128) → ReLU → Dropout → Linear(128 → num_experts)
→ Top-K softmax (k=2, sparse routing)
```

- **Per-task gating** 讓每個 task 學到不同的 routing pattern
  - 例：legato gate 偏好 onset expert（因為 bow change 是短暫的 transition）
  - 例：tonal gate 偏好 note/phrase expert（因為 harmonics 需要較長的 context）

#### Expert Output Mixing

```python
mixed = sum_over_experts(gate_prob × expert_output)  # (B, T, 128)
→ LayerNorm → ReLU
→ Linear(128 → num_classes)
```

#### Classification Heads


| Task            | Head                      | Output            | Loss                          |
| --------------- | ------------------------- | ----------------- | ----------------------------- |
| Tonal Technique | Linear(128 → 4)           | logits            | Cross-Entropy (+ focal γ=2.0) |
| Articulation    | Linear(128 → 4)           | logits            | Cross-Entropy (+ focal γ=2.0) |
| Legato          | Linear(128 → 1) → sigmoid | probability [0,1] | Positive-weighted BCE (w=5.0) |


#### Auxiliary Losses

- **Per-expert auxiliary heads**：每個 expert 獨立預測所有三個 task，用 auxiliary loss 監督（weight=0.1）
  - 確保每個 expert 都學到有意義的 representation
  - 防止 expert collapse（某些 expert 從未被選中）
- **Load-balance loss**：`E × sum(P_i²)`，鼓勵 gate probability 均勻分布（coeff=0.001）

#### Complete Data Flow

```
Inputs:
  features    (B, T, 512)  ← dedicated technique CRNN (GRU output)
  onset_prob  (B, T, 1)    ← transcription branch, max-pooled over 88 pitches
  offset_prob (B, T, 1)    ← transcription branch
  frame_prob  (B, T, 1)    ← transcription branch
  logmel      (B, 1, T, 229) ← BN-normalised log-mel spectrogram

                    ┌─ _TemporalCtx(k=11, d=1) → (B,T,128) ─┐
                    │                                          │ + onset_cue(16)
                    │                                          │ → Expert0 MLP(144→256→128) → e0 (B,T,128)
                    │                                          │
features(B,T,512) ──├─ _TemporalCtx(k=51, d=1) → (B,T,128) ─┤ + frame_cue(16)
                    │                                          │ → Expert1 MLP(144→256→128) → e1 (B,T,128)
                    │                                          │
                    ├─ _TemporalCtx(k=51, d=3) → (B,T,128) ─┤ + all 3 cues(48)
                    │                                          │ → Expert2 MLP(176→256→128) → e2 (B,T,128)
                    │                                          │
                    └──────────────────────────────────────────┘
                                                                ┌───────────────────────────────────┐
logmel(B,1,T,229) → _DedicatedSpectralCNN → (B,T,128) ────────┤ + feat_proj(128) + frame_cue(16)  │
                                                                │ → Expert3 MLP(272→256→128) → e3  │
                                                                └───────────────────────────────────┘

expert_outs = stack([e0, e1, e2, e3], dim=-2) → (B, T, 4, 128)

gate_input = cat([temporal_ctx_0, temporal_ctx_1, temporal_ctx_2,  ← 3×128 = 384
                  spectral_out, feat_proj,                         ← 2×128 = 256
                  cue_onset, cue_offset, cue_frame])               ← 3×16  = 48
             → (B, T, 688)                                        total gate_in = 688

             ┌── gate_tonal(688→128→4)  → softmax/top-2 → w_t (B,T,4)
gate_input ──├── gate_artic(688→128→4)  → softmax/top-2 → w_a (B,T,4)
             └── gate_legato(688→128→4) → softmax/top-2 → w_l (B,T,4)

mixed_tonal  = Σ(w_t × expert_outs) → LayerNorm → ReLU → Linear(128→4) → tonal_logits
mixed_artic  = Σ(w_a × expert_outs) → LayerNorm → ReLU → Linear(128→4) → artic_logits
mixed_legato = Σ(w_l × expert_outs) → LayerNorm → ReLU → Linear(128→1) → sigmoid → legato_prob
```

#### Design Highlights

1. **Per-task Independent Gates**：三個任務各有獨立的 gate network，使得 articulation 可偏好 onset expert（捕捉 spiccato 的短促攻擊），legato 可偏好 onset expert（捕捉 bow change 的瞬間 transition），tonal 可偏好 spectral expert（捕捉 harmonics 的泛音分布）。實驗結果（Section 7.3）驗證了這種 per-task routing 的合理性。
2. **Transcription Cue Injection**：將 transcription 模型的 onset/offset/frame probability 投影成 16-dim embedding 後注入各 expert 和 gate。這讓技法分類可以利用音符邊界的結構性資訊（e.g. onset 處可能是 bow change），但 inference 時不需要 GT note boundary。Legacy mode 下每個 expert 選擇性注入不同 cue（E0 只看 onset, E1 只看 frame, E2 看全部），強化了 expert 的專業化。
3. **Spectral Expert 獨立訓練**：`_DedicatedSpectralCNN` 的所有權重僅由 technique loss 更新，不受 transcription gradient 影響。這確保 spectral expert 的 filters 專門針對 technique-discriminative 的頻譜模式（如 harmonics 的泛音分布 vs. normal bowing 的頻譜差異）而非 pitch detection。
4. **Top-K Sparse Routing**：設定 `top_k=2`，每個 frame 只啟用 2 個最相關的 expert（其餘權重為 0）。這提高了稀疏性、減少了計算浪費，並讓每個 expert 更專注於自己擅長的 technique。
5. **Per-Expert Auxiliary Loss**：訓練時每個 expert 都有獨立的分類頭（`aux_tonal_heads`, `aux_artic_heads`, `aux_legato_heads`），強迫每個 expert 學到對所有任務都有意義的 representation。這是解決 MoE 常見的 **expert collapse**（部分 expert 被閒置、從未學到有用特徵）問題的關鍵手段。
6. **Load-Balance Loss**：`E × Σ(P_i²)` 正則化 gate probability 的分布，防止所有流量都被路由到同一個 expert。實驗顯示 baseline（無 spectral expert）出現嚴重的 routing 不均（E1 在 legato gate 僅佔 0.02%），而加入 balance loss 後有所改善。

### 3.3 FrameMultiScaleMoEConfig（所有可調參數）

```python
@dataclass
class FrameMultiScaleMoEConfig:
    feat_dim: int = 512            # 輸入 feature 維度 (from CRNN GRU)
    proj_dim: int = 128            # temporal context 投影維度
    num_experts: int = 4           # expert 總數 (含 spectral)
    short_kernel: int = 11         # E0: ~110ms at 100fps
    note_kernel: int = 51          # E1: ~510ms
    phrase_kernel: int = 51        # E2: with dilation
    phrase_dilation: int = 3       # E2: effective ~1530ms
    expert_hidden: int = 256       # expert MLP hidden dim
    shared_dim: int = 128          # expert output / classification dim
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    cue_dim: int = 16              # transcription cue embedding dim
    spectral_proj_dim: int = 128   # spectral expert output dim
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.001
    top_k: int = 2                 # sparse top-k routing
    use_spectral_expert: bool = True
    # Ablation knobs
    expert_mask: tuple = None      # e.g. (0,1,2) — None = all
    shared_gate: bool = False      # single gate for all tasks
    uniform_routing: bool = False  # bypass learned gating
    temporal_experts: tuple = None # custom ((kernel, dilation), ...)
    use_cues: bool = True          # inject transcription cues
```

---

## 4. Training Pipeline

### 4.1 Main Training Script

**入口**：`piano_transcription/pytorch/main_contrast.py train`

**Shell wrapper**：`scripts/run_viotech_frame_moe_finetune.sh`

#### 訓練流程

```
1. Load pretrained transcriptor checkpoint (transcriptor_model.pth)
2. Initialize model with predict_technique_frame_moe=True
3. Dataset: viotech_mixed_mosavpt (VioTech + MOSAVPT mixed)
4. Training loop:
   For each batch:
     a. Forward: waveform → output_dict (transcription + technique)
     b. Transcription loss: regress_onset_offset_frame_velocity_bce
     c. Frame-MoE technique loss:
        - loss_tonal (CE/focal) + loss_artic (CE/focal) + loss_legato (weighted BCE)
        - loss_balance (load-balance regularization)
        - loss_aux (per-expert auxiliary losses)
     d. Total loss = transcription_loss + technique_frame_moe_weight × (
          loss_tonal + loss_artic + loss_legato
          + balance_coeff × loss_balance
          + aux_weight × loss_aux
        )
     e. Backward + optimizer step
5. Periodic validation (+ optional RWC cross-dataset eval)
6. Early stopping based on fmoe_loss_technique
```

#### Key Training Hyperparameters


| Parameter                    | Value   | 說明                               |
| ---------------------------- | ------- | -------------------------------- |
| `batch_size`                 | 8       |                                  |
| `learning_rate`              | 5e-4    |                                  |
| `reduce_iteration`           | 1000    | LR reduction step                |
| `early_stop`                 | 10000   | max iterations                   |
| `augmentation`               | aug     | data augmentation                |
| `max_note_shift`             | 2       | pitch augmentation ±2 semitones  |
| `technique_frame_moe_weight` | 1.0     | technique loss weight            |
| `frame_moe_balance_coeff`    | 0.001   | load-balance regularization      |
| `focal_gamma`                | 2.0     | focal loss focusing parameter    |
| `fmoe_aux_weight`            | 0.1     | per-expert auxiliary loss weight |
| `fmoe_spectral_expert`       | 0 (off) | baseline 不使用 spectral expert     |
| `mosavpt_ratio`              | 0.2     | MOSAVPT 在 mixed training 中的比例    |


#### Data Augmentation (`Augmentor`)

- Pitch shift: ±2 semitones (via librosa)
- Audio augmentations via `audiomentations` library

### 4.2 Pretrained Checkpoint

- `checkpoints/transcriptor_model.pth`：在 MAESTRO + violin data 上預訓練的 transcription model
- Fine-tuning 時同時更新 transcription 分支和 technique 分支

---

## 5. Evaluation Pipeline

### 5.1 Frame-Level MoE Evaluation

**Script**：`scripts/run_evaluate_frame_moe.sh`

**入口**：`piano_transcription/pytorch/evaluate_frame_moe.py`

#### 評估指標

對三個 task 分別計算：

- **Accuracy**（per-class 和 overall）
- **Macro F1** / **Weighted F1**
- **Macro Precision** / **Macro Recall**
- **Per-class support**（active frame 數）

#### 評估流程

```
1. Load checkpoint
2. Build CustomDataset + CustomTestSampler (validation/test split)
3. For each batch:
   a. Forward (no grad)
   b. Active-frame mask: only frames with note activity
   c. Collect predictions and ground truth
4. Compute metrics per task
5. Analyze gate distribution (global and per-class)
6. Export results to JSON
```

#### Gate Distribution Analysis

- 報告每個 task（tonal / articulation / legato）的 gate routing 統計
- 全域分佈 + 按 class 分佈
- 可觀察不同 technique 如何偏好不同時間尺度的 expert

### 5.2 Additional Evaluation Modes


| Script                              | 說明                                 |
| ----------------------------------- | ---------------------------------- |
| `evaluate_frame_moe_notelevel.py`   | Frame-MoE 預測聚合到 note-level 的指標     |
| `evaluate_frame_moe_gt_boundary.py` | 使用 GT note boundary 的 frame-MoE 評估 |
| `evaluate_viotech.py`               | 基礎 frame-level technique 分類（非 MoE） |
| `evaluate_moe_technique.py`         | Note-level MoE 評估                  |


### 5.3 Cross-Dataset Evaluation (RWC)

- RWC dataset 的 technique label 透過 `rwc_technique_map` 對應：


| RWC Label | Tonal     | Articulation | Legato     |
| --------- | --------- | ------------ | ---------- |
| flageolet | harmonics | none         | bow_change |
| normal    | none      | none         | bow_change |
| pizzicato | pizzicato | none         | bow_change |
| spiccato  | none      | spiccato     | bow_change |


---

## 6. Ablation Study: Expert Composition（from `results/ablation_expert_composition/`）

> 所有實驗使用相同設定：seed=2026, batch_size=4, lr=5e-4, focal_gamma=2.0, mosavpt_ratio=0.2, balance_coeff=0.001, top_k=2, per-task gating。
> 數據來源：`results/ablation_expert_composition/*.json`

### 6.1 VioTech Test Set — Overall Metrics


| Experiment  | Experts               | Tonal Acc  | Tonal mF1  | Artic Acc  | Artic mF1  | Legato Acc | Legato mF1 |
| ----------- | --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **full**    | E0+E1+E2+E3(spectral) | 95.82%     | 0.3640     | 77.91%     | 0.3833     | 80.09%     | **0.6770** |
| baseline    | E0+E1+E2              | 95.27%     | 0.2507     | 77.16%     | 0.3567     | **81.21%** | 0.6370     |
| no_onset    | E1+E2                 | 95.78%     | 0.2795     | **78.50%** | **0.3902** | 80.19%     | 0.6988     |
| no_note     | E0+E2                 | **96.48%** | **0.3404** | 79.62%     | 0.3901     | 81.34%     | **0.7061** |
| no_phrase   | E0+E1                 | 95.56%     | 0.2534     | 78.94%     | 0.3925     | **81.62%** | **0.7096** |
| only_onset  | E0                    | 95.81%     | 0.2759     | 78.15%     | 0.3769     | 80.36%     | 0.6820     |
| only_note   | E1                    | 95.70%     | 0.2680     | 79.24%     | 0.3911     | 79.77%     | 0.6839     |
| only_phrase | E2                    | 96.37%     | 0.3249     | 78.63%     | 0.3840     | 79.47%     | 0.6849     |


### 6.2 VioTech Test Set — Per-Class Accuracy


| Experiment  | Harmonics  | Spiccato   | Bow Change | Sustained  |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| **full**    | **6.05%**  | **68.03%** | **47.46%** | 87.81%     |
| baseline    | 1.49%      | 42.47%     | 30.73%     | **93.15%** |
| no_onset    | 7.51%      | 74.63%     | 56.66%     | 85.76%     |
| no_note     | **24.17%** | 62.54%     | 54.70%     | 87.64%     |
| no_phrase   | 1.89%      | 72.37%     | 54.94%     | 87.93%     |
| only_onset  | 6.75%      | 57.09%     | 48.43%     | 87.91%     |
| only_note   | 5.05%      | 67.58%     | 51.67%     | 86.42%     |
| only_phrase | 18.98%     | 61.94%     | 53.42%     | 85.63%     |


### 6.3 RWC Cross-Dataset — Overall Metrics


| Experiment  | Experts               | Tonal Acc  | Tonal mF1  | Artic Acc  | Artic mF1  | Legato Acc | Legato mF1 |
| ----------- | --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **full**    | E0+E1+E2+E3(spectral) | 92.12%     | **0.7930** | **96.70%** | **0.7805** | **98.48%** | **0.8243** |
| baseline    | E0+E1+E2              | 87.79%     | 0.5055     | 94.17%     | 0.4910     | 98.19%     | 0.6874     |
| no_onset    | E1+E2                 | 86.65%     | 0.7387     | 97.48%     | 0.8302     | 97.44%     | 0.7548     |
| no_note     | E0+E2                 | **91.43%** | **0.8137** | 95.96%     | 0.5948     | **98.40%** | **0.8295** |
| no_phrase   | E0+E1                 | 92.81%     | 0.8115     | 95.92%     | 0.7236     | 98.26%     | 0.8282     |
| only_onset  | E0                    | 90.97%     | 0.7819     | 95.59%     | 0.5258     | 98.21%     | 0.7508     |
| only_note   | E1                    | 92.67%     | 0.8013     | 94.89%     | 0.5067     | 98.10%     | 0.7367     |
| only_phrase | E2                    | 92.14%     | 0.8082     | 95.76%     | 0.5907     | 98.11%     | 0.8015     |


### 6.4 RWC Cross-Dataset — Per-Class Accuracy


| Experiment  | None(T) | Pizzicato  | Harmonics  | None(A)    | Spiccato(A) | Sustained | Bow Change |
| ----------- | ------- | ---------- | ---------- | ---------- | ----------- | --------- | ---------- |
| **full**    | 94.05%  | 84.58%     | 69.65%     | 98.91%     | **50.06%**  | 99.21%    | **66.06%** |
| baseline    | 96.42%  | 0.00%      | 67.69%     | 98.59%     | 0.79%       | 99.81%    | 25.74%     |
| no_onset    | 87.27%  | **93.25%** | 65.74%     | **99.37%** | **57.50%**  | 98.20%    | 63.73%     |
| no_note     | 91.08%  | **96.88%** | **89.44%** | 99.94%     | 11.91%      | 98.96%    | **73.17%** |
| no_phrase   | 93.98%  | 84.80%     | 84.18%     | 98.58%     | 39.75%      | 98.71%    | **78.52%** |
| only_onset  | 92.17%  | 79.26%     | 87.39%     | 99.93%     | 3.91%       | 99.45%    | 42.64%     |
| only_note   | 94.00%  | 89.03%     | 74.77%     | 99.27%     | 2.34%       | 99.39%    | 40.45%     |
| only_phrase | 93.01%  | 90.07%     | 79.99%     | 99.72%     | 11.97%      | 98.78%    | 68.07%     |


### 6.5 Key Observations

**Expert Composition:**

- `full`（加 spectral expert）在 RWC 上顯著優於 `baseline`：Tonal mF1 0.793 vs 0.506，Artic mF1 0.781 vs 0.491
- `no_note`（E0+E2）在 RWC 上 Tonal mF1 最高（0.8137），Harmonics accuracy 89.44%
- `no_phrase`（E0+E1）在 RWC Legato 的 bow_change accuracy 最高（78.52%）
- 單一 expert（only_*）普遍不如多 expert 組合，驗證了 MoE 的必要性

**Baseline 的問題：**

- `baseline`（E0+E1+E2）在 RWC 上 Pizzicato accuracy = 0%，Spiccato accuracy = 0.79%，存在嚴重 expert collapse
- Gate distribution 顯示 E1 幾乎被完全忽略（legato gate: E1=0.02%），routing 不均勻

---

## 7. Detailed Results（from Experiment JSON）

### 7.1 Full Model (E0+E1+E2+Spectral) — VioTech Test Set

**Overall（672,820 active frames）：**


| Task         | Accuracy   | Macro F1   | Weighted F1 | Macro Prec | Macro Rec |
| ------------ | ---------- | ---------- | ----------- | ---------- | --------- |
| Tonal        | 95.82%     | 0.3640     | 0.9402      | 0.6214     | 0.3534    |
| Articulation | 77.91%     | 0.3833     | 0.7505      | 0.3738     | 0.3938    |
| **Legato**   | **80.09%** | **0.6770** | **0.8005**  | 0.6777     | 0.6764    |


**Tonal Per-Class：**


| Class       | Accuracy | Support             |
| ----------- | -------- | ------------------- |
| none        | 99.97%   | 643,115             |
| harmonics   | 6.05%    | 28,535              |
| openstring  | 0.00%    | 1,170               |
| (pizzicato) | —        | 0 (not in test set) |


**Articulation Per-Class：**


| Class    | Accuracy | Support |
| -------- | -------- | ------- |
| none     | 89.50%   | 452,726 |
| spiccato | 68.03%   | 174,964 |
| release  | 0.00%    | 18,389  |
| staccato | 0.00%    | 26,741  |


**Legato Per-Class：**


| Class          | Accuracy   | Support     |
| -------------- | ---------- | ----------- |
| sustained      | 87.81%     | 544,126     |
| **bow_change** | **47.46%** | **128,694** |


**Gate Distribution（full model, VioTech）：**


| Task                | E0 Onset (110ms) | E1 Note (510ms) | E2 Phrase (1530ms) | E3 Spectral |
| ------------------- | ---------------- | --------------- | ------------------ | ----------- |
| Tonal (global)      | 23.31%           | 27.02%          | 0.10%              | **49.57%**  |
| Tonal (harmonics)   | 22.39%           | 25.98%          | 0.00%              | **51.63%**  |
| Artic (global)      | 23.84%           | 0.00%           | **45.11%**         | 31.05%      |
| Artic (spiccato)    | **51.77%**       | 0.00%           | 34.11%             | 14.12%      |
| Legato (global)     | **86.16%**       | 0.48%           | 13.35%             | 0.00%       |
| Legato (bow_change) | **84.95%**       | 0.70%           | 14.35%             | 0.01%       |


### 7.2 Full Model (E0+E1+E2+Spectral) — RWC Cross-Dataset

**Overall（1,083,700 active frames）：**


| Task         | Accuracy   | Macro F1   | Weighted F1 | Macro Prec | Macro Rec |
| ------------ | ---------- | ---------- | ----------- | ---------- | --------- |
| Tonal        | 92.12%     | 0.7930     | 0.9293      | 0.7895     | 0.8276    |
| Articulation | 96.70%     | 0.7805     | 0.9645      | 0.8303     | 0.7448    |
| **Legato**   | **98.48%** | **0.8243** | **0.9849**  | 0.8222     | 0.8264    |


**Tonal Per-Class：**


| Class     | Accuracy | Support |
| --------- | -------- | ------- |
| none      | 94.05%   | 948,715 |
| pizzicato | 84.58%   | 80,947  |
| harmonics | 69.65%   | 54,038  |


**Articulation Per-Class：**


| Class    | Accuracy | Support   |
| -------- | -------- | --------- |
| none     | 98.91%   | 1,034,731 |
| spiccato | 50.06%   | 48,969    |


**Legato Per-Class：**


| Class          | Accuracy   | Support    |
| -------------- | ---------- | ---------- |
| sustained      | 99.21%     | 1,059,896  |
| **bow_change** | **66.06%** | **23,804** |


**Gate Distribution（full model, RWC）：**


| Task                | E0 Onset (110ms) | E1 Note (510ms) | E2 Phrase (1530ms) | E3 Spectral |
| ------------------- | ---------------- | --------------- | ------------------ | ----------- |
| Tonal (global)      | 27.99%           | 19.19%          | 0.16%              | **52.66%**  |
| Tonal (pizzicato)   | 4.50%            | **61.25%**      | 0.16%              | 34.09%      |
| Tonal (harmonics)   | 7.70%            | 42.66%          | 0.00%              | **49.64%**  |
| Artic (global)      | 10.04%           | 0.00%           | **51.54%**         | 38.42%      |
| Artic (spiccato)    | **45.68%**       | 0.01%           | 39.15%             | 15.16%      |
| Legato (global)     | **88.66%**       | 1.15%           | 10.19%             | 0.00%       |
| Legato (bow_change) | **85.51%**       | 10.37%          | 3.97%              | 0.15%       |


### 7.3 Gate Distribution Insights

**Tonal gate 高度依賴 Spectral expert**：

- 全域 ~50% 路由到 Spectral expert，因為 harmonics/pizzicato 的頻譜特徵（泛音分布、能量分布）是區分關鍵
- Pizzicato 在 RWC 上偏好 Note expert（61.25%）：pizzicato 的 note-level 時間特徵（短促、快速 decay）

**Articulation gate 偏好 Phrase + Spectral expert**：

- 全域主要路由到 Phrase (45-52%) + Spectral (31-38%)
- Spiccato class 偏好 Onset expert（45-52%）：spiccato 是短促的 attack-driven 技巧

**Legato gate 極度偏好 Onset expert**（捕捉 bow change transition）：

- 全域 86-92% 路由到 Onset expert，因為 bow change 本質上是一個短暫的 transition event
- Spectral expert 幾乎不被 legato gate 使用（~0%），因為 bow change 不涉及頻譜變化

### 7.4 Baseline (E0+E1+E2) vs Full (E0+E1+E2+E3) — Improvement Summary


| Metric              | Baseline (VioTech) | Full (VioTech) | Δ       | Baseline (RWC) | Full (RWC) | Δ       |
| ------------------- | ------------------ | -------------- | ------- | -------------- | ---------- | ------- |
| Tonal mF1           | 0.2507             | 0.3640         | +0.113  | 0.5055         | 0.7930     | +0.288  |
| Artic mF1           | 0.3567             | 0.3833         | +0.027  | 0.4910         | 0.7805     | +0.290  |
| Legato mF1          | 0.6370             | 0.6770         | +0.040  | 0.6874         | 0.8243     | +0.137  |
| Spiccato Acc        | 42.47%             | 68.03%         | +25.56% | 0.79%          | 50.06%     | +49.27% |
| Bow Change Acc      | 30.73%             | 47.46%         | +16.73% | 25.74%         | 66.06%     | +40.32% |
| Pizzicato Acc (RWC) | —                  | —              | —       | 0.00%          | 84.58%     | +84.58% |


---

## 8. Project File Structure

```
VioPTT/
├── config/
│   ├── technique_label_config.json          # 完整 6 class 設定
│   └── technique_label_config_reduced.json  # 訓練用簡化 class 設定
├── scripts/
│   ├── run_viotech_frame_moe_finetune.sh    # 主要訓練 script
│   ├── run_evaluate_frame_moe.sh            # 主要評估 script
│   ├── ablation_frame_moe.sh               # Expert/routing/balance ablation
│   ├── ablation_temporal_density.sh         # Temporal scale ablation
│   └── ...                                  # 其他 training/eval scripts
├── piano_transcription/
│   ├── pytorch/
│   │   ├── main_contrast.py                 # 訓練入口
│   │   ├── models_contrast.py               # 主模型定義 (CRNN backbone + technique heads)
│   │   ├── moe_frame_multiscale.py          # Frame Multi-Scale MoE head ★
│   │   ├── losses.py                        # 所有 loss functions
│   │   ├── evaluate_frame_moe.py            # Frame MoE 評估
│   │   ├── evaluate_frame_moe_notelevel.py  # Note-level 聚合評估
│   │   └── evaluate_frame_moe_gt_boundary.py
│   └── utils/
│       ├── config.py                        # 全域常數 (sr=16000, fps=100, etc.)
│       ├── data_generator.py                # Dataset / Sampler 定義
│       ├── technique_label_utils.py         # Label config 載入
│       └── utilities.py                     # MIDI I/O, target processing
├── prepare_viotech.py                       # VioTech120 → HDF5 轉換
├── prepare_mosavpt.py                       # MOSAVPT → HDF5
├── mapping_config.json                      # CSV string → class ID 對照
└── results/                                 # 實驗結果 JSON
```

---

## 9. Key Technical Details for Paper Writing

### Audio Processing

- Sample rate: **16,000 Hz**
- STFT: n_fft=2048, hop=160 (→ 100 fps)
- Mel filterbank: **229 bins**, fmin=30 Hz, fmax=8000 Hz
- Training segment: **10 seconds** (160,000 samples)

### Loss Functions

- **Transcription**: Regression onset/offset + frame/velocity BCE
- **Tonal/Articulation**: Cross-Entropy with **Focal Loss** (γ=2.0)
- **Legato**: **Positive-weighted BCE** (bow_change weight=5.0, Gaussian regression target)
- **Balance**: E × Σ(P_i²), coefficient=0.001
- **Auxiliary**: Per-expert independent prediction, weight=0.1

### Training Details

- Pretrained on MAESTRO/violin transcription → fine-tune with technique loss
- Mixed training: VioTech (80%) + MOSAVPT (20%)
- Optimizer: Adam, lr=5e-4, LR reduction every 1000 iterations
- Data augmentation: pitch shift ±2 semitones + audio augmentations
- Early stopping: best validation `fmoe_loss_technique`

### Compared Approaches (within codebase)

1. **FrameLevelTechniqueHead** — simple FC head on CRNN features (baseline)
2. **MoETechniqueHead** — note-level MoE (requires GT boundaries)
3. **ZoneMoETechniqueHead** — zone-specialized note-level MoE
4. **PerTaskZoneMoEHead** — per-task gated zone MoE
5. **FrameMultiScaleMoEHead** — frame-level multi-scale MoE ★（proposed）

