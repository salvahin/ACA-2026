# Prompts para Generación de Imágenes - TC3002B-2026

> **Total de imágenes identificadas: ~170**
>
> Este documento contiene todos los prompts optimizados para Nano Banana 2 organizados por módulo.
> Para cada imagen se incluye: ubicación exacta, tipo, descripción y prompt listo para copiar.

---

## Tabla de Contenidos

1. [Módulo AI](#módulo-ai) - 23 imágenes
2. [Módulo Compilers](#módulo-compilers) - 30 imágenes
3. [Módulo Stats](#módulo-stats) - 28 imágenes
4. [Módulo Research](#módulo-research) - 35 imágenes
5. [Módulo Project_1](#módulo-project_1) - 27 imágenes
6. [Módulo Project_2](#módulo-project_2) - 27 imágenes

---

# Módulo AI

## 01_IA_Clasica_vs_Generativa.md

### Imagen AI-1.1: Timeline Evolución de IA
- **Ubicación**: `lecturas/AI/01_IA_Clasica_vs_Generativa.md:84`
- **Tipo**: Timeline / Evolución histórica
- **Descripción**: Timeline visual mostrando los 4 eras principales (Sistemas Expertos, ML Clásico, Deep Learning, LLMs)

```
Timeline + Evolution of AI + Four eras (Expert Systems 1960s, Classical ML 1990s, Deep Learning 2010s, LLMs 2020s) + Each era shows: timeline position, key characteristics, example algorithms + Minimalist design with clear horizontal progression + Labels and annotations for each epoch
```

---

### Imagen AI-1.2: Discriminativo vs Generativo
- **Ubicación**: `lecturas/AI/01_IA_Clasica_vs_Generativa.md:100`
- **Tipo**: Diagrama comparativo
- **Descripción**: Visualización lado a lado: discriminativo como línea de separación, generativo como distribución de probabilidad

```
Comparison diagram + Discriminative vs Generative models + Left side: discriminative showing decision boundary separating cats and dogs in 2D feature space with line + Right side: generative showing probability distribution curve over data + Use contrasting colors (blue for discriminative, purple for generative) + Include sample points and clear labels + Mathematical notation optional but visual emphasis on concepts
```

---

### Imagen AI-1.3: Arquitectura Red Neuronal
- **Ubicación**: `lecturas/AI/01_IA_Clasica_vs_Generativa.md:155`
- **Tipo**: Arquitectura de red neuronal
- **Descripción**: Visualización de red neuronal con capas de entrada, ocultas y salida

```
Neural network architecture + Simple feedforward network + 3 input nodes, 2-3 hidden layers with varying nodes, 1 output node + Each connection shows weight parameter (w) + Activation function icons at each layer (ReLU, sigmoid) + Use spheres/circles for neurons and lines for connections + Color gradient from input (green) through hidden (blue) to output (red) + Minimalist style with clear node positioning
```

---

### Imagen AI-1.4: Retropropagación 4 Pasos
- **Ubicación**: `lecturas/AI/01_IA_Clasica_vs_Generativa.md:168`
- **Tipo**: Proceso/flujo conceptual
- **Descripción**: Flujo visual de los 4 pasos: forward pass, error calculation, backward pass, weight update

```
Process flowchart + Backpropagation algorithm + 4 main steps: 1) Forward pass arrow right with data, 2) Error calculation with loss function symbol, 3) Backward pass arrow left with gradient symbols, 4) Weight update with adjustment arrows + Color code steps (green→red→blue→orange) + Use directional arrows clearly + Include mathematical symbols but keep visual primary + Top section shows forward, bottom shows backward + Annotations for each step describing the computation
```

---

### Imagen AI-1.5: RNN vs Transformer
- **Ubicación**: `lecturas/AI/01_IA_Clasica_vs_Generativa.md:195`
- **Tipo**: Comparación arquitectónica
- **Descripción**: RNN procesamiento secuencial vs Transformer atención paralela

```
Comparison diagram + RNN vs Transformer architecture + Top half: RNN with 4 words processed sequentially with arrows flowing left to right and recurrent feedback loops (dashed arrows) + Bottom half: Transformer with 4 words with all-to-all attention connections (no sequential arrows, all connections shown simultaneously as network) + Color code: RNN in blue sequential theme, Transformer in purple parallel theme + Include time indicators (t=1,2,3,4) for RNN, simultaneous for Transformer + Annotations explaining parallel vs sequential advantages
```

---

## 02_Fundamentos_Deep_Learning.md

### Imagen AI-2.1: Funciones de Activación
- **Ubicación**: `lecturas/AI/02_Fundamentos_Deep_Learning.md:36`
- **Tipo**: Gráficas matemáticas comparativas
- **Descripción**: 4 gráficas de funciones (Sigmoid, Tanh, ReLU, GELU)

```
Comparison plot grid + Four activation functions + 2x2 grid with Sigmoid, Tanh, ReLU, GELU + Each subplot: x-axis from -5 to 5, y-axis showing output range + Draw smooth curves for Sigmoid (S-shape 0-1), Tanh (S-shape -1 to 1), ReLU (linear 0 and flat for negatives), GELU (smooth approximation of ReLU) + Color each function distinctly + Add horizontal reference lines at 0, 0.5, 1 + Label output ranges on y-axis + Include mathematical function symbol in corner of each subplot + Grid background light, curves thick and visible
```

---

### Imagen AI-2.2: Red Neuronal Multicapa con Tensores
- **Ubicación**: `lecturas/AI/02_Fundamentos_Deep_Learning.md:117`
- **Tipo**: Diagrama arquitectónico
- **Descripción**: Red con dimensiones de tensores en cada paso

```
Neural network architecture diagram + Multi-layer network with tensor dimensions + Input layer showing tensor shape (32, 784) + Layer 1: W (784x128), ReLU, output (32, 128) + Layer 2: W (128x64), ReLU, output (32, 64) + Output layer: W (64x10), Softmax, output (32, 10) + Show each layer as rectangular block with dimension labels + Arrows between layers + Color code: input green, processing blue, output red + Include operation symbols (⊕ for addition, matrix multiply symbol) + Show batch dimension (32) conceptually through stacking
```

---

### Imagen AI-2.3: Propagación del Gradiente
- **Ubicación**: `lecturas/AI/02_Fundamentos_Deep_Learning.md:188`
- **Tipo**: Diagrama de flujo
- **Descripción**: Cómo el error fluye hacia atrás, gradientes disminuyendo

```
Flowchart + Error backpropagation through layers + Network architecture showing 4 layers horizontal + Forward pass arrows pointing right (thicker, green) at top + Backward pass arrows pointing left (thicker, red) at bottom + Layer outputs labeled with L1, L2, L3, L4 + Gradient symbols flowing backward with arrows showing magnitude (dashes indicate smaller gradients) + Vanishing gradient concept: gradient size decreases left to right in backward pass + Color intensity shows gradient magnitude + Include loss symbol at output layer + Annotations for chain rule at one layer
```

---

### Imagen AI-2.4: Mecanismo de Atención
- **Ubicación**: `lecturas/AI/02_Fundamentos_Deep_Learning.md:246`
- **Tipo**: Comparación conceptual
- **Descripción**: RNN vs Transformer con énfasis en atención ponderada

```
Attention mechanism illustration + RNN vs Transformer focus + Top: RNN showing 5 words with sequential processing (boxes connected in chain, each fed to hidden state) + Bottom: Transformer showing same 5 words with attention matrix (all-to-all connections shown as weighted arrows with opacity indicating attention weight) + Highlight one word (e.g., "saltó") with attention weights to all other words shown as colored connections (darker=higher attention) + Left side annotations explaining sequential bottleneck + Right side annotations explaining parallel processing + Use color gradient (light to dark) to show attention weights
```

---

## 03_Generacion_Autoregresiva.md

### Imagen AI-3.1: Softmax Transformación
- **Ubicación**: `lecturas/AI/03_Generacion_Autoregresiva.md:60`
- **Tipo**: Transformación matemática
- **Descripción**: Logits → Softmax → Probabilidades

```
Mathematical transformation diagram + Softmax function visualization + Left side: input logits as bars with values [2.3, 0.5, 8.1, 7.9, 6.2] (unrestricted range, some negative) + Middle: Softmax function box with formula softmax(x) = e^x / Σe^x + Right side: output probabilities as bars [0.15, 0.02, 0.65, 0.12, 0.06] (all positive, sum to 1) + Color first bar (smallest) in blue, last bar (largest) in red to show transformation + Include arrow with "Normalization" label + Add reference lines at 0 on left, at sum=1 on right + Show mathematical notation above transformation
```

---

### Imagen AI-3.2: Estrategias de Sampling
- **Ubicación**: `lecturas/AI/03_Generacion_Autoregresiva.md:117`
- **Tipo**: Comparación visual
- **Descripción**: Greedy vs Top-K vs Top-P

```
Sampling strategies comparison + Token probability distribution comparison + Show probability distribution bars [0.35, 0.32, 0.18, 0.07, 0.05, 0.03] three times side by side + Greedy: one bar highlighted (0.35 token 0) + Top-K (K=5): five bars highlighted with annotation showing renormalization [0.37, 0.34, 0.19, 0.07, 0.03] + Top-P (P=0.9): bars until sum exceeds 0.9 highlighted [0.35, 0.32, 0.18, 0.07] with cumulative sum annotation + Use color: selected bars green, unselected bars gray + Add renormalization visualization under each strategy + Include sampling action (dart/selection indicator) on chosen subset + Annotations explaining tradeoff between diversity and quality
```

---

### Imagen AI-3.3: Beam Search
- **Ubicación**: `lecturas/AI/03_Generacion_Autoregresiva.md:213`
- **Tipo**: Árbol de búsqueda
- **Descripción**: 3 beams expandiéndose con probabilidades

```
Tree diagram + Beam search exploration + Root node "START" at left + Three vertical lines representing time steps (t=1, t=2, t=3) + First column: 3 initial tokens (beams) "el", "gato", "saltó" with probabilities 0.35, 0.32, 0.18 + Arrows expanding from each beam to 2-3 next tokens + Second column: combinations like "el gato" (0.14), "el perro" (0.10), "gato saltó" (0.16) + Third column: top 3 selected beams highlighted + Use color gradient for probabilities (darker=higher) + Include probability values on edges and nodes + Show pruning operation (X marks on low-probability branches) + Bold highlight on final selected beams
```

---

### Imagen AI-3.4: BPE Paso a Paso
- **Ubicación**: `lecturas/AI/03_Generacion_Autoregresiva.md:327`
- **Tipo**: Algoritmo paso a paso
- **Descripción**: Byte Pair Encoding fusionando caracteres

```
Step-by-step algorithm visualization + Byte Pair Encoding process + Initial vocabulary shown: [a, a, b, a, a, a, b] as separate blocks + Step 1: Count pairs, highlight "aa" pair with frequency 3 + Step 2: Merge "aa" to "X", new vocabulary [X, b, X, X, b] + Step 3: Count pairs again, highlight "Xb" frequency 2 + Step 4: Merge "Xb" to "Y", new vocabulary [Y, X, Y] + Step 5: Continue one more iteration + Color new merged tokens distinctly + Show frequency counts above each vocabulary state + Use arrows labeled "merge" between steps + Include final vocabulary size annotation + Minimize text, maximize visual representation of token fusion
```

---

## 04_Arquitectura_Transformer.md

### Imagen AI-4.1: Query-Key-Value
- **Ubicación**: `lecturas/AI/04_Arquitectura_Transformer.md:70`
- **Tipo**: Diagrama conceptual
- **Descripción**: Cómo Query se compara con Keys usando Values

```
Attention mechanism diagram + Query-Key-Value concept + Top: word "saltó" with query vector (Q) shown as arrow pointing right + Middle: all words ("el", "gato", "saltó", "cerca") with key vectors (K) as vertical arrows + Connections from Q to each K shown as lines with similarity scores (dotted lines thicker for higher similarity) + Scores labeled: 0.1 for "el", 2.5 for "gato", 0.0 for "saltó", 1.8 for "cerca" + Below each word: value vectors (V) as colored arrows + Final aggregation showing weighted sum of values with attention weights + Color scheme: Q in red, K in blue, V in green, similarity lines in orange + Include softmax symbol over scores + Annotations for each component
```

---

### Imagen AI-4.2: Multi-Head Attention
- **Ubicación**: `lecturas/AI/04_Arquitectura_Transformer.md:163`
- **Tipo**: Diagrama arquitectónico
- **Descripción**: 3 cabezas de atención capturando diferentes patrones

```
Multi-head attention architecture + Sentence example "el perro saltó sobre la cerca" shown at top + 3 horizontal rows representing 3 attention heads + Head 1 (Subject-verb): show attention weights from "saltó" to "perro" (high), "el" (medium), others (low) with curved connections + Head 2 (Object): show attention weights from "saltó" to "cerca" (high), "sobre" (medium) + Head 3 (Context): show attention weights distributed across multiple words + Each head shown with different color (blue, green, orange) + Below: concatenation and projection step showing combination of 3 heads + Use matrix-like notation for attention weights (heat map style) + Include legend explaining what each head captures
```

---

### Imagen AI-4.3: Positional Encoding
- **Ubicación**: `lecturas/AI/04_Arquitectura_Transformer.md:194`
- **Tipo**: Heatmap 2D
- **Descripción**: Patrón sinusoidal de PE(pos, dim)

```
Positional encoding heatmap + 2D visualization with dimensions × positions + X-axis: sequence positions (0-15) + Y-axis: embedding dimensions (0-63) + Color intensity shows PE value from -1 (dark blue) to +1 (dark red) + Visible sine/cosine wave patterns at different frequencies (low frequency waves on top/bottom, high frequency in middle) + Include frequency labels on right side + Small chart on side showing PE formula with sin and cos components + Add annotations explaining why different frequencies help model learn positional relationships + Grid lines to show position and dimension boundaries
```

---

### Imagen AI-4.4: Bloque Transformer
- **Ubicación**: `lecturas/AI/04_Arquitectura_Transformer.md:272`
- **Tipo**: Arquitectura / Bloque funcional
- **Descripción**: Anatomía completa de un bloque Transformer

```
Transformer block architecture diagram + Input tensor enters from left + Multi-Head Attention box in center-top + Output arrows to Add layer (residual connection shown as bypass arrow from input) + LayerNorm box + Feed-Forward Network (FFN) box with expansion (3072) and reduction back (768) + Another Add layer + Another LayerNorm + Output arrow to right + Use color: input green, attention blue, norm yellow, FFN red, output green + Show skip connections (curved arrows around blocks) clearly + Include dimension information at each step + Residual path shown as alternative path (dashed lines) + Include mathematical notation (+ for addition, norm symbol) minimal but present + Layer normalization and FFN functions labeled with key operations
```

---

### Imagen AI-4.5: Encoder vs Decoder
- **Ubicación**: `lecturas/AI/04_Arquitectura_Transformer.md:290`
- **Tipo**: Comparación arquitectónica
- **Descripción**: BERT (bidireccional) vs GPT (causal)

```
Encoder vs Decoder architecture comparison + Left side: BERT Encoder with 12 stacked blocks labeled "Encoder Block 1-12" + Sentence "el gato saltó" shown at bottom with all words + Attention connections between all words shown as full mesh (all-to-all) highlighted in blue + Annotations: "Bidirectional: can see all context" + Right side: GPT Decoder with 12 stacked blocks + Same sentence shown at bottom + Attention connections showing only left triangular pattern (position 0 can see position 0, position 1 can see positions 0-1, position 2 can see 0-2) highlighted in red + Masked connections shown with X marks or blocked arrows + Annotations: "Causal: can only see past and current" + Include legend showing bidirectional vs causal masking + Bottom note explaining why each approach + Use contrasting colors (blue for encoder, red for decoder)
```

---

## 05_BERT_GPTs_Tokenizacion.md

### Imagen AI-5.1: Arquitectura BERT
- **Ubicación**: `lecturas/AI/05_BERT_GPTs_Tokenizacion.md:40`
- **Tipo**: Diagrama arquitectónico
- **Descripción**: BERT con [CLS], tokens, [SEP] y 12 capas

```
BERT architecture diagram + Input sequence at bottom: "[CLS] El banco está cerca del río [SEP]" with token boxes + Embeddings layer showing token embedding + segment embedding + positional encoding combined + Stacked 12 blocks labeled "Transformer Layer 1-12" + Each layer shows multi-head attention icon + Feed-forward icon + Layer norm symbols + Arrows flowing upward through layers + Output at top showing contextualized representations + Color: input orange, embedding layers yellow, transformer blocks blue, output green + Include dimension information (768 hidden, 12 heads) as annotations + Show special tokens [CLS], [SEP] distinctly + Include legend for layer components
```

---

### Imagen AI-5.2: GPT Scaling Laws
- **Ubicación**: `lecturas/AI/05_BERT_GPTs_Tokenizacion.md:82`
- **Tipo**: Gráfico de escalado
- **Descripción**: Relación entre tamaño y capacidades emergentes

```
GPT scaling and capabilities emergence + X-axis: Model size in billions of parameters (log scale 1B to 1000B) + Y-axis: Capability level (from 1 to 10) + Plotted points: GPT-1 (117M), GPT-2 (1.5B), GPT-3 (175B), GPT-4 (1.7T) positioned horizontally + Curve showing general capability improvement (S-shaped curve) + Different colored regions/bands for: Text completion, Few-shot learning, Chain-of-thought, Complex reasoning + Annotations pointing to each GPT version with year + Emergent capability thresholds marked with dashed lines and labeled + Model sizes labeled on points + Include note about phase transitions + Color gradient from bottom (simple tasks) to top (complex reasoning) + Area under curve highlight to show scaling law
```

---

### Imagen AI-5.3: Estrategias de Tokenización
- **Ubicación**: `lecturas/AI/05_BERT_GPTs_Tokenizacion.md:154`
- **Tipo**: Comparación visual
- **Descripción**: Word-level vs Character-level vs Subword

```
Tokenization strategies comparison + Three columns showing word "unforgettable" + Column 1 (Word-level): "unforgettable" as single token (1 token) with vocabulary size 100K+ annotation + Column 2 (Character-level): u-n-f-o-r-g-e-t-t-a-b-l-e (13 tokens) with vocabulary size 256 annotation + Column 3 (Subword BPE): "un" "forget" "table" (3 tokens) with vocabulary size 30K annotation + Below each column: metrics showing token count, vocab size, OOV rate + Use color: word in green (1 block), characters in red (13 blocks), subwords in blue (3 blocks) + Add problem/advantage labels under each: "huge vocab/OOV", "long sequences", "balanced" + Include example unknown word handling for each strategy + Visualization of vocabulary sizes as bar chart on right side
```

---

### Imagen AI-5.4: Comparativa Tokenizadores
- **Ubicación**: `lecturas/AI/05_BERT_GPTs_Tokenizacion.md:187`
- **Tipo**: Tabla comparativa visual
- **Descripción**: BPE vs WordPiece vs SentencePiece

```
Comparison table visualization + Create 3x4 grid showing BPE, WordPiece, SentencePiece + Rows: Algorithm, Merge Strategy, Prefix Handling, Used In + Row 1 Algorithm: BPE shows "frequency", WordPiece shows "likelihood", SentencePiece shows "unigram/BPE" + Row 2 Merge: show examples of merges for each + Row 3 Prefix: "none" vs "##" vs "▁" symbols shown + Row 4 Used In: "GPT/CodeX", "BERT", "T5/LLaMA" + Use distinct colors for each tokenizer (columns different colors) + Include visual icons for each concept + Add brief explanation icon that shows tooltip info + Use clear cell borders + Include small example output for each tokenizer showing how it tokenizes same input
```

---

## 06_Sampling_Constrained_Decoding.md

### Imagen AI-6.1: Pipeline Constrained Decoding
- **Ubicación**: `lecturas/AI/06_Sampling_Constrained_Decoding.md:109`
- **Tipo**: Pipeline de proceso
- **Descripción**: Logits → Máscara → Distribución restringida

```
Constrained decoding pipeline diagram + Step 1: Logits bar chart showing unrestricted values [2.1, 0.5, 3.2, 1.1, -5.2, ...] + Step 2: Valid tokens mask represented as checkbox grid (checked for valid, unchecked for invalid) + Step 3: Masking operation showing logits of invalid tokens changed to -∞ (visual representation: changed to dark red) + Step 4: Softmax transformation showing probability distribution with zero probabilities for masked tokens + Step 5: Final sampling showing only valid tokens with restored probabilities + Arrow flowing right through steps showing transformation + Color: valid tokens green checkmarks, invalid red X marks + Include equations at key steps (mask operation, softmax, probability) + Show example: JSON token stream with only valid JSON tokens allowed + Bottom annotation explaining guarantee: "Output respects constraints 100%"
```

---

### Imagen AI-6.2: JSON State Machine
- **Ubicación**: `lecturas/AI/06_Sampling_Constrained_Decoding.md:195`
- **Tipo**: Máquina de estados
- **Descripción**: Estados y transiciones de parser JSON

```
State machine diagram + JSON parsing states as circles: "Start" → "Object" → "Key" → "Colon" → "Value" → "Comma" → loop back + Each state labeled with valid tokens in that state + Arrows between states labeled with token triggers (e.g., "{" triggers State start → Object) + Color states by category: "Start" in green (entry), "Key/Value" in blue (main), "Comma" in orange (transition), "End" in red (final) + Include invalid token handling: arrows pointing to error state + Show example sequence "{ " name " : " John " , "age" : 25 }" with colored path through states + Include text annotations explaining each state transition + Add a table showing valid tokens for each state
```

---

### Imagen AI-6.3: Trade-off Calidad vs Restricciones
- **Ubicación**: `lecturas/AI/06_Sampling_Constrained_Decoding.md:323`
- **Tipo**: Gráfico de trade-off
- **Descripción**: Relación inversa entre restricciones y calidad

```
Trade-off curve visualization + X-axis: Constraint Strictness (0=none, 100=full) + Y-axis: Output Quality / Semantic Quality (0-100) + Curved line showing inverse relationship (starts high at 0 constraint, decreases as constraints increase) + Shaded area above curve labeled "Ideal outputs excluded by constraints" + Mark three points on curve: "No restrictions" (0,95), "Moderate JSON grammar" (40,88), "Strict format" (100,75) + Annotations: "Reintention cost" at high quality low constraint side, "Guarantee validity" at low quality high constraint side + Add cost indicators: low cost at high constraint, high cost at low constraint (inverse curve below) + Color zones: green for balanced sweet spot, red for overly restricted + Include specific example in each zone + Legend explaining what the gap represents
```

---

## 07_Fine_Tuning_Evaluacion.md

### Imagen AI-7.1: LoRA Decomposición
- **Ubicación**: `lecturas/AI/07_Fine_Tuning_Evaluacion.md:122`
- **Tipo**: Diagrama arquitectónico
- **Descripción**: Matriz W y LoRA como A @ B de bajo rango

```
LoRA architecture decomposition + Left: Original weight matrix W (large, 4096x4096) shown as large square + Center: LoRA decomposition showing ΔW = A @ B where A (4096x8) and B (8x4096) + Show relative sizes: W as large square 100%, A as tall thin rectangle labeled "4096x8", B as short wide rectangle "8x4096" + Calculation: A @ B = smaller matrix multiplication shown with arrows + Right: Updated weights W' = W + ΔW shown as superposition + Color: W in light blue (frozen), A in orange, B in green, ΔW in purple, W' in dark blue + Include dimension labels prominently + Add comparison: "16M parameters → 64K parameters" (256x reduction) + Show this applied to Q,V projection matrices with multiple instances + Bottom note: "Red dot for update-only, blue for original weights" + Include speedup metric "50x training speedup"
```

---

### Imagen AI-7.2: Métricas de Evaluación
- **Ubicación**: `lecturas/AI/07_Fine_Tuning_Evaluacion.md:287`
- **Tipo**: Comparación de métricas
- **Descripción**: BLEU, ROUGE, F1, EM evaluando misma salida

```
Evaluation metrics comparison visualization + Reference text: "The cat is black" at top + Generated text: "The cat is black" directly below (perfect match) + Metrics shown as gauges/bars: BLEU (100%), ROUGE-L (100%), Exact Match (100%), F1 Score (100%) + All gauges show 100% (green full) + Second comparison below: + Reference: "The cat is black" + Generated: "The black cat is very beautiful" + Metrics shown: BLEU (50%), ROUGE-L (60%), Exact Match (0%), F1 Score (45%) + Gauges partially filled in orange/yellow + Third comparison: + Reference: "Paris is the capital of France" + Generated: "The capital of France is Paris" + Metrics: BLEU (70%), ROUGE-L (85%), Exact Match (0%), F1 Score (80%) + Mixed gauges showing different metric behaviors + Use color gradient (red=low, yellow=medium, green=high) + Include explanation of what each metric captures + Show which metrics work best for different scenarios (box/label on side)
```

---

### Imagen AI-7.3: Pass@k Metrics
- **Ubicación**: `lecturas/AI/07_Fine_Tuning_Evaluacion.md:360`
- **Tipo**: Gráfico de distribución
- **Descripción**: Probabilidad de éxito con más intentos

```
Pass@k metric visualization + X-axis: Number of attempts (1, 5, 10, 100) + Y-axis: Success rate percentage (0-100%) + Bar chart showing Pass@1 (70%), Pass@5 (85%), Pass@10 (90%), Pass@100 (92%) + Bars increasing height left to right + Above each bar: show example with that many generated samples + Pass@1: single solution code block + Pass@5: 5 solution code blocks stacked with success checkmark on correct one + Pass@10: 10 blocks with checkmark on one + Pass@100: abstract representation of 100 blocks with one checkmark + Color bars green (growing success) + Add annotation showing diminishing returns (gap between 5→10 smaller than 1→5) + Include curve showing theoretical improvement pattern + Bottom: probability formula Pass@k ≈ 1 - (1 - Pass@1)^k + Example numbers: "If 70% of first attempts work, then 85% chance at least 1 of 5 works"
```

---

### Imagen AI-7.4: Benchmark Contamination
- **Ubicación**: `lecturas/AI/07_Fine_Tuning_Evaluacion.md:408`
- **Tipo**: Diagrama de flujo
- **Descripción**: Tipos de contaminación en benchmark

```
Benchmark contamination types diagram + Three parallel flows showing contamination paths + Path 1: Pre-training data → includes benchmark → Model sees during pre-training → learns answers → High benchmark score (misleading) + Path 2: Fine-tuning data → includes benchmark → Model memorizes → High benchmark score (false improvement) + Path 3: Training data ≠ benchmark but similar distribution → Model overfits to training → Similar benchmark data shows artificially high performance + Use red arrows for contamination paths + Green checkmark for clean flow at bottom (clean training) + Each path shows: data source → model training → benchmark evaluation → false result highlight + Include "Actual generalization" vs "Apparent generalization" comparison + Add detection methods in boxes: hash matching, embedding similarity, manual inspection + Color contamination paths in red, clean path in green + Include percentage estimates of performance inflation from each source
```

---

## 08_MLOps_Visualizacion.md

### Imagen AI-8.1: MLOps Tracking Components
- **Ubicación**: `lecturas/AI/08_MLOps_Visualizacion.md:33`
- **Tipo**: Diagrama de componentes
- **Descripción**: Code, Data, Config, Metrics, Artifacts interconectados

```
MLOps tracking components diagram + Central hub labeled "Experiment Tracking" + Five surrounding nodes: Code, Data, Configuration, Metrics, Artifacts + Each node shows key information: + Code: git commit hash, dependencies list + Data: dataset name, splits percentages, version + Config: hyperparameters list (lr, batch size, etc) + Metrics: accuracy, loss curve, time + Artifacts: model file, logs, predictions + Connecting lines from each node to center showing data flow + Color code: Code blue, Data green, Config orange, Metrics red, Artifacts purple + Icons for each category + Include wandb/MLflow logos as tracking tools + Add annotation: "Without tracking: chaos, without tracking: reproducibility" with before/after comparison + Show single experiment version linking to all five components
```

---

### Imagen AI-8.2: Training Curves
- **Ubicación**: `lecturas/AI/08_MLOps_Visualizacion.md:313`
- **Tipo**: Gráfico de líneas
- **Descripción**: Train vs Validation loss con overfitting

```
Training curves example plot + Two curves on same plot with epochs on X-axis (0-100) and loss on Y-axis (0-5) + Blue curve (Train loss): smoothly decreasing from 4.5 at epoch 0 to 0.1 at epoch 100 + Orange curve (Validation loss): decreases from 4.5 to minimum 0.8 around epoch 40, then increases to 2.1 at epoch 100 + Annotations: "Good fit region" showing area where both curves are close + "Overfitting region" showing area where train < val + Vertical dashed line at epoch 40 labeled "Early stopping point" + Add shaded area highlighting overfitting zone + Color train curve dark blue, validation curve orange + Include legend + Add annotations for learning insights: "Model starting to overfit here", "optimal stopping point" + Grid lines with light gray + Title "Training Progress: Loss vs Epoch"
```

---

### Imagen AI-8.3: Heatmap de Resultados
- **Ubicación**: `lecturas/AI/08_MLOps_Visualizacion.md:351`
- **Tipo**: Heatmap/matriz
- **Descripción**: Accuracy variando temperatura y nivel de gramática

```
Heatmap of results grid + X-axis: Grammar Level (L1-L2, L1-L3, L1-L4) + Y-axis: Temperature (T=0.1, T=0.5, T=1.0) + 3x3 grid with values: + [0.85, 0.82, 0.79] for T=0.1 + [0.80, 0.78, 0.75] for T=0.5 + [0.72, 0.70, 0.68] for T=1.0 + Color intensity from blue (0.68) to red (0.85) showing performance + Values annotated in each cell + Color bar on right showing scale from 0.6 (dark blue) to 0.9 (dark red) + Title "Correctness Rate by Configuration" + Include interpretation notes: "Higher temperature = more creativity but less accuracy", "Stricter grammar = lower accuracy but guaranteed format" + Highlight best configuration with border/star + Gridlines between cells
```

---

### Imagen AI-8.4: Scatter con Tendencia
- **Ubicación**: `lecturas/AI/08_MLOps_Visualizacion.md:376`
- **Tipo**: Gráfico de dispersión
- **Descripción**: Tokens generados vs tiempo de ejecución

```
Scatter plot with trend line + X-axis: Tokens Generated (100-500) + Y-axis: Execution Time (ms) (200-2000) + ~30 scatter points distributed in positive correlation pattern + Points colored by density (lighter sparse, darker dense areas) + Linear trend line (red dashed) through points showing relationship + Include confidence interval shading around trend line (light red area) + Correlation coefficient R² value shown in corner (e.g., "R² = 0.87") + Axes labels with units + Title "Generation Cost vs Token Count" + Add annotations: "1 token ≈ 30ms" at one point + Include outliers discussion annotation + Grid in light gray + Legend showing points and trend line + Y-axis baseline reference at 0 time + Use scatter point alpha transparency to show overlapping points
```

---

# Módulo Compilers

## 01-introduccion-compiladores.md

### Imagen COMP-1.1: Pipeline de Compilación
- **Ubicación**: `lecturas/Compilers/01-introduccion-compiladores.md:24`
- **Tipo**: Diagrama de flujo/pipeline
- **Descripción**: Pipeline completo: Lexing → Parsing → Análisis Semántico → Optimización → Codegen

```
Compiler pipeline flowchart + Five stages of compilation + Elements: Lexing (input characters, output tokens), Parsing (input tokens, output AST), Semantic Analysis (input AST, output validated AST), Optimization (input code, output optimized code), Codegen (input optimized code, output executable machine code) + Vertical flow style with boxes and arrows, clean separation between stages + Labels for each stage with data flow annotations
```

---

### Imagen COMP-1.2: Jerarquía de Chomsky
- **Ubicación**: `lecturas/Compilers/01-introduccion-compiladores.md:219`
- **Tipo**: Diagrama de pirámide
- **Descripción**: 4 niveles: Tipo 0, Tipo 1, Tipo 2, Tipo 3

```
Chomsky hierarchy pyramid + Four language types + Elements: Tipo 0 (Recursively enumerable, unrestricted rules), Tipo 1 (Context-sensitive, αAβ → αγβ), Tipo 2 (Context-free, A → β), Tipo 3 (Regular, A → aB) + Pyramid structure with decreasing expressiveness from top to bottom, arrows showing subset relationships + Color gradient from complex to simple, labels showing power and limitations
```

---

### Imagen COMP-1.3: Máquinas y Tipos de Chomsky
- **Ubicación**: `lecturas/Compilers/01-introduccion-compiladores.md:200`
- **Tipo**: Diagrama de correspondencia
- **Descripción**: Qué máquina reconoce cada tipo de Chomsky

```
Chomsky types to automata mapping + Language theory associations + Elements: Type 0 with Turing Machine icon, Type 1 with Linear Bounded Automaton, Type 2 (Context-free) with Pushdown Automaton, Type 3 with Finite State Automaton + Four columns with vertical layout, arrows connecting types to machines + Clean visual separation with technology icons, descriptive labels
```

---

## 02-lenguajes-regulares-dfas.md

### Imagen COMP-2.1: DFA "hola"
- **Ubicación**: `lecturas/Compilers/02-lenguajes-regulares-dfas.md:82`
- **Tipo**: Autómata finito determinista
- **Descripción**: DFA que reconoce "hola" con 5 estados

```
DFA finite automaton + Recognizing string "hola" + Elements: Five states (q0 initial, q1 after h, q2 after o, q3 after l, q4 final/accept), transitions labeled with characters 'h', 'o', 'l', 'a' + Circles for states, double circle for accept state, labeled arrows for transitions + Horizontal left-to-right layout showing character consumption sequence
```

---

### Imagen COMP-2.2: DFA Binarios con Ciclo
- **Ubicación**: `lecturas/Compilers/02-lenguajes-regulares-dfas.md:106`
- **Tipo**: Autómata con ciclos
- **Descripción**: DFA que acepta strings de 0s y 1s

```
DFA binary numbers + Non-deterministic transitions creating loops + Elements: Initial state q0, accepting state q1, self-loop on q1 for 0 and 1 symbols, transition from q0 to q1 on 0 and 1 + Circles for states (double circle for accept), curved arrows for transitions including self-loop + Compact layout showing repetition capability with clear loop indicator
```

---

### Imagen COMP-2.3: DFA Identificadores
- **Ubicación**: `lecturas/Compilers/02-lenguajes-regulares-dfas.md:138`
- **Tipo**: Autómata con clases de caracteres
- **Descripción**: DFA que reconoce identificadores [a-zA-Z_][a-zA-Z0-9_]*

```
DFA identifier validation + Character class transitions + Elements: q0 initial state, q1 accepting state, transition q0 to q1 on [a-zA-Z_], self-loop on q1 for [a-zA-Z0-9_], error transitions for invalid characters + Circles and double circle for final state, character class labels in brackets + Clear visual distinction between accepting and rejecting paths
```

---

## 03-nfas-conversion-dfa.md

### Imagen COMP-3.1: NFA "gato"|"gatos"
- **Ubicación**: `lecturas/Compilers/03-nfas-conversion-dfa.md:40`
- **Tipo**: Autómata no determinista
- **Descripción**: NFA que acepta "gato" O "gatos"

```
NFA non-deterministic automaton + Ambiguous path alternatives + Elements: Initial state q0, intermediate states q1-q4 for g-a-t-o sequence, q5 for optional s, two accepting states (q4 for "gato", q5 for "gatos"), transitions showing non-determinism + Circles for states, double circles for multiple accept states, branching arrows showing choice points, dotted style for epsilon transitions
```

---

### Imagen COMP-3.2: Subset Construction
- **Ubicación**: `lecturas/Compilers/03-nfas-conversion-dfa.md:195`
- **Tipo**: Transformación algorítmica
- **Descripción**: Pasos de conversión NFA a DFA

```
NFA to DFA conversion visualization + Subset construction algorithm + Elements: Original NFA states on left showing a* then b pattern, right side showing DFA with subset states {q0,q2}, {q1,q2}, {q3}, arrows showing how states merge through subset construction + Two side-by-side automata with transformation arrows, numbered steps showing progression, merged state indicators
```

---

### Imagen COMP-3.3: Epsilon Clausura
- **Ubicación**: `lecturas/Compilers/03-nfas-conversion-dfa.md:145`
- **Tipo**: Diagrama de alcanzabilidad
- **Descripción**: Cómo calcular ε-clausura

```
Epsilon closure calculation graph + Transitive epsilon transitions + Elements: States q0, q1, q2, q3, q4 with epsilon transitions between them, highlighted closure set for each state starting from q0 showing {q0,q1,q2,q3} + Curved epsilon-labeled arrows showing reachability, state subsets highlighted with backgrounds, visual grouping of closure sets
```

---

## 04-gramaticas-libres-contexto.md

### Imagen COMP-4.1: Parse Tree Expresión
- **Ubicación**: `lecturas/Compilers/04-gramaticas-libres-contexto.md:162`
- **Tipo**: Árbol de análisis sintáctico
- **Descripción**: Parse tree para "2 + 3 * 4"

```
Parse tree for arithmetic expression + Operator precedence hierarchy + Elements: Root node E (expression), left child E (value 2), operator + node, right child T (term) with children T*F (3*4), showing 3 and 4 as leaves, demonstrating that multiplication groups tighter than addition + Tree structure with root at top, terminals as leaves, indentation and hierarchy showing precedence, labeled nodes with grammar symbols
```

---

### Imagen COMP-4.2: Ambigüedad en Gramáticas
- **Ubicación**: `lecturas/Compilers/04-gramaticas-libres-contexto.md:217`
- **Tipo**: Comparación de dos árboles
- **Descripción**: Dos parse trees para "2 + 3 * 4" (20 vs 14)

```
Grammar ambiguity comparison + Same string, different interpretations + Elements: Two complete parse trees for "2+3*4" side by side, left tree showing (2+3)*4 evaluation order, right tree showing 2+(3*4), both starting from same expression symbol but with different structures + Dual parse trees with clear labeling of which produces 20 and which 14, arrows showing evaluation order, different colors for each tree
```

---

### Imagen COMP-4.3: Jerarquía de Precedencia
- **Ubicación**: `lecturas/Compilers/04-gramaticas-libres-contexto.md:258`
- **Tipo**: Diagrama de capas
- **Descripción**: Jerarquía de operadores en gramática

```
Grammar precedence hierarchy + Operator precedence layers + Elements: Top level for addition/subtraction (E → E+T | T), middle level for multiplication/division (T → T*F | F), bottom level for factors and parentheses (F → (E) | number), showing how structure encodes precedence + Pyramid or tiered layout showing decreasing precedence upward, grammar rules annotated on each level, terminals at bottom
```

---

## 05-bnf-ebnf-parsing.md

### Imagen COMP-5.1: Top-Down vs Bottom-Up
- **Ubicación**: `lecturas/Compilers/05-bnf-ebnf-parsing.md:177`
- **Tipo**: Comparativa de dos enfoques
- **Descripción**: Árbol crece hacia abajo vs hacia arriba

```
Top-down vs bottom-up parsing comparison + Dual tree growth directions + Elements: Left side shows top-down with expression at root, terms below, tokens at leaves (top to bottom); Right side shows bottom-up with tokens at bottom, reductions building up to root + Two complete parse trees growing in opposite directions, annotated with "predictive/top-down" on left and "shift-reduce/bottom-up" on right, arrows showing direction of analysis
```

---

### Imagen COMP-5.2: LL(1) vs LR(1)
- **Ubicación**: `lecturas/Compilers/05-bnf-ebnf-parsing.md:207`
- **Tipo**: Tabla de transiciones
- **Descripción**: Comparación de tablas de parsing

```
LL vs LR parsing decision tables + Parser state matrices + Elements: LL(1) table showing states and single lookahead decisions, LR(1) table with action/goto separations, stack-based transitions + Two side-by-side parsing tables, LL showing simple lookups, LR showing shift/reduce actions, different table structures highlighted, annotations on capabilities and grammar restrictions
```

---

### Imagen COMP-5.3: Earley Parser
- **Ubicación**: `lecturas/Compilers/05-bnf-ebnf-parsing.md:248`
- **Tipo**: Visualización de algoritmo
- **Descripción**: Chart-based parsing con predicción, scanning, completación

```
Earley parser algorithm visualization + Chart-based parsing process + Elements: Input tokens across top, chart positions 0 to n with items/dotted rules, showing prediction (inactive items), scanning (consuming tokens), completion (forming higher-level items) + Grid-like structure with rows as chart entries, columns as parse positions, items shown with dots indicating position, arrows showing item progression
```

---

## 06-pipeline-fsm-xgrammar.md

### Imagen COMP-6.1: Pipeline XGrammar 10 Fases
- **Ubicación**: `lecturas/Compilers/06-pipeline-fsm-xgrammar.md:40`
- **Tipo**: Pipeline completo
- **Descripción**: Todas las fases desde especificación hasta parser

```
XGrammar compilation pipeline + Ten-stage grammar to code transformation + Elements: Parsing → Normalization → NFA Construction → Epsilon Elimination → Subset Construction → DFA Minimization → Rule Inlining → Dead Code Elimination → Lookahead Analysis → Parser Generation, with data between stages + Vertical flowchart with 10 boxes, arrows showing data flow, intermediate representations shown (CFG→NFA→DFA→Parser), annotations on what happens at each stage
```

---

### Imagen COMP-6.2: Thompson NFA Construction
- **Ubicación**: `lecturas/Compilers/06-pipeline-fsm-xgrammar.md:108`
- **Tipo**: Construcción de autómatas
- **Descripción**: Concatenación, alternancia, repetición

```
Thompson NFA construction + Regular expression to automaton + Elements: Three patterns shown - sequence (a concatenated with b), alternation (a OR b with epsilon transitions), Kleene star (a* with loop and epsilon bypass) + Three separate automata showing construction rules, epsilon transitions shown as dotted lines, clear marking of initial and final states
```

---

### Imagen COMP-6.3: Token Mask Cache
- **Ubicación**: `lecturas/Compilers/06-pipeline-fsm-xgrammar.md:371`
- **Tipo**: Diagrama de optimización
- **Descripción**: Bitmap de tokens válidos, O(1) lookup

```
Adaptive token mask cache optimization + Bitwise token validation + Elements: Parser state with associated token mask (bitmap of valid tokens), example with 8 tokens showing binary representation 01010101, comparison of naive lookup vs O(1) bitmap operation + State node with attached bitmap visualized as bits, equation showing bitwise operation for checking validity, performance comparison annotations
```

---

## 07-diseno-gramaticas-dsl.md

### Imagen COMP-7.1: Construcción Incremental
- **Ubicación**: `lecturas/Compilers/07-diseno-gramaticas-dsl.md:24`
- **Tipo**: Fases progresivas
- **Descripción**: 5 iteraciones de desarrollo de gramática

```
Incremental grammar design phases + Bottom-up development progression + Elements: Phase 1 (basic numbers), Phase 2 (addition), Phase 3 (multiplication with precedence), Phase 4 (parentheses), Phase 5 (variables), each showing production rules + Vertical stack or steps showing progression, each phase building on previous, example inputs/outputs for each phase, test cases passing at each level
```

---

### Imagen COMP-7.2: Torre de Precedencia
- **Ubicación**: `lecturas/Compilers/07-diseno-gramaticas-dsl.md:112`
- **Tipo**: Pirámide de niveles
- **Descripción**: 10 niveles de operadores

```
Operator precedence hierarchy in grammar + Ten-level operator tower + Elements: Level 1 (assignment), Level 2 (OR), Level 3 (AND), Level 4 (comparison), Level 5 (addition/subtraction), Level 6 (multiplication/division), Level 7 (exponentiation), Level 8 (unary), Level 9 (indexing), Level 10 (atoms) + Pyramid structure with decreasing precedence from top, grammar rules shown for each level, operators listed with their level
```

---

### Imagen COMP-7.3: Left-Factoring
- **Ubicación**: `lecturas/Compilers/07-diseno-gramaticas-dsl.md:296`
- **Tipo**: Antes/después
- **Descripción**: Problema de backtracking y solución

```
Left factoring optimization + Common prefix extraction + Elements: Before side showing statement rules for assignment/increment/decrement with identifier prefix repeatedly used, After side showing factored version with statement_rest alternatives, LL(1) lookahead analysis highlighted + Two-column comparison, left side showing inefficient version with duplication, right side showing factored clean version, arrows indicating transformation, lookahead symbol boxes showing decision clarity
```

---

## 08-compilacion-gramaticas.md

### Imagen COMP-8.1: Greibach Normal Form
- **Ubicación**: `lecturas/Compilers/08-compilacion-gramaticas.md:19`
- **Tipo**: Transformación de gramática
- **Descripción**: Conversión a GNF

```
Greibach Normal Form transformation + Terminal-first grammar conversion + Elements: Original grammar with mixed recursion, Converted GNF with every rule starting with terminal (A → aα pattern), shows equivalence, example with expr and term rules before and after + Two grammar boxes with transformation arrow, highlighting terminals at start of rules, explanatory text on benefits for GPU parsing prediction, comparison of rule count
```

---

### Imagen COMP-8.2: Table Compression
- **Ubicación**: `lecturas/Compilers/08-compilacion-gramaticas.md:119`
- **Tipo**: Optimización de tabla
- **Descripción**: State merging para reducir tamaño

```
Parsing table compression + State merging optimization + Elements: Original larger transition table with duplicate rows (q1 and q2 identical), after merging showing compressed table with fewer states, visual highlighting of identical rows, size reduction annotation + Before/after table comparison, rows with same transitions highlighted, merged state representation, memory savings calculated and shown
```

---

### Imagen COMP-8.3: AST Construction
- **Ubicación**: `lecturas/Compilers/08-compilacion-gramaticas.md:314`
- **Tipo**: Árbol con estructura de datos
- **Descripción**: Parse tree → AST

```
Abstract syntax tree construction + Grammar rule to data structure + Elements: Complete parse tree for expression, corresponding AST with data structure representation (Expr {left: Term, ops: [Op(...)]}), simplified node types, elimination of intermediate grammar symbols + Left side shows detailed parse tree, right side shows AST, transformation arrows showing how parse tree contracts to AST, code representation shown alongside tree
```

---

## 09-limitaciones-context-sensitive.md

### Imagen COMP-9.1: Jerarquía Chomsky Completa
- **Ubicación**: `lecturas/Compilers/09-limitaciones-context-sensitive.md:35`
- **Tipo**: Diagrama completo
- **Descripción**: 4 tipos con ejemplos y máquinas

```
Chomsky hierarchy complete + All four types with examples and automata + Elements: Type 0 (Recursively enumerable, turing machine, unrestricted rules), Type 1 (Context-sensitive, linear bounded automaton, context matters), Type 2 (Context-free, pushdown automaton, a^n b^n example), Type 3 (Regular, DFA/NFA, identifiers example) + Pyramid structure, each level showing language type, example string, grammar rule format, and automaton type, decreasing complexity downward
```

---

### Imagen COMP-9.2: Por qué CFG NO puede a^n b^n c^n
- **Ubicación**: `lecturas/Compilers/09-limitaciones-context-sensitive.md:66`
- **Tipo**: Visualización de limitación
- **Descripción**: CFG con 1 pila falla contra 3 contadores

```
CFG limitation visualization + Why context-free cannot count three items + Elements: PDA with single stack trying to match a^n b^n c^n, showing how second counter (b's) corrupts first counter (a's) on stack, example derivation showing intermediate invalid strings + Stack visualization for failed match attempt, showing stack underflow/overflow, comparison with how Type 1 would need multiple independent counters, labeled annotation of why LIFO limitation fails
```

---

### Imagen COMP-9.3: Multi-Pass Analysis
- **Ubicación**: `lecturas/Compilers/09-limitaciones-context-sensitive.md:361`
- **Tipo**: Pipeline de 4 pasadas
- **Descripción**: Lexing → Parsing → Semántico → Codegen

```
Multi-pass compiler architecture + Layered analysis beyond CFG + Elements: Pass 1 Lexing (chars to tokens), Pass 2 Parsing CFG (tokens to AST), Pass 3 Semantic Analysis (symbol table, type checking, scope), Pass 4 Optimization/Codegen, showing data flow and validation at each stage + Four horizontal boxes with downward flow, intermediate representations shown, AST transformations highlighted, semantic analysis detail expanded
```

---

## 10-parser-generators-reflexion.md

### Imagen COMP-10.1: Comparativa Herramientas
- **Ubicación**: `lecturas/Compilers/10-parser-generators-reflexion.md:268`
- **Tipo**: Matriz de comparación
- **Descripción**: ANTLR vs Tree-Sitter vs Bison vs Earley vs XGrammar

```
Parser generator tools comparison + Decision matrix for tool selection + Elements: Five tools (ANTLR, Tree-Sitter, Bison, Earley, XGrammar) compared on: performance, ease of use, grammar generality, error recovery, community size + Matrix layout with tools as rows, characteristics as columns, color coding for strengths/weaknesses, speed/quality indicators, use-case recommendations annotated
```

---

### Imagen COMP-10.2: Arquitectura XGrammar
- **Ubicación**: `lecturas/Compilers/10-parser-generators-reflexion.md:212`
- **Tipo**: Diagrama de arquitectura
- **Descripción**: Gramática → Procesamiento → Parser + GPU validator

```
XGrammar architecture diagram + Specialized GPU compilation path + Elements: Input specification, XGrammar compiler with subphases (parse, validate, optimize, lookahead, generate), dual output: CPU parser and GPU kernel validator + Box diagram showing input on left, XGrammar processor in middle with internal stages, two output paths on right (CPU executable and GPU kernel), GPU-specific optimizations highlighted
```

---

### Imagen COMP-10.3: Compilador Perfecto
- **Ubicación**: `lecturas/Compilers/10-parser-generators-reflexion.md:389`
- **Tipo**: Pipeline conceptual
- **Descripción**: 5 fases ideales de compilador

```
Ideal compiler architecture + Five-phase synthesis + Elements: Phase 1 Lexing (DFA/flex), Phase 2 Parsing (CFG/ANTLR), Phase 3 Semantic Analysis (custom), Phase 4 Optimization (custom), Phase 5 Codegen (custom), showing tool choices and responsibilities + Horizontal pipeline with five major phases, showing what's tool-generated vs hand-coded, data representations at each boundary, tool recommendations for each phase
```

---

# Módulo Stats

## 01-Fundamentos-Probabilidad.md

### Imagen STATS-1.1: Diagrama de Venn
- **Ubicación**: `lecturas/Stats/01-Fundamentos-Probabilidad.md:25`
- **Tipo**: Diagrama conceptual
- **Descripción**: Unión, Intersección y Complemento de eventos

```
Three overlapping Venn diagrams showing event operations + Set theory visualization + Left: Union A∪B (both circles colored), Middle: Intersection A∩B (overlap colored), Right: Complement A^c (outside A colored) + Labeled circles A, B, and universal set Ω + Clean, educational style with distinct colors (blue for A, red for B, green for results) + Text annotations: "A∪B", "A∩B", "A^c" + Mathematics textbook quality
```

---

### Imagen STATS-1.2: Tabla de Contingencia
- **Ubicación**: `lecturas/Stats/01-Fundamentos-Probabilidad.md:82`
- **Tipo**: Heatmap
- **Descripción**: P(Compiló | Restricciones) vs P(Compiló | Baseline)

```
Contingency table heatmap with 2x2 layout + Dataset: 1000 kernel compilations (Baseline vs Restricciones, Compiled vs Not) + Rows: "Baseline" and "Restricciones", Columns: "Compiló" and "No compiló" + Cell values: 750, 250, 850, 150 with percentages + Color intensity representing cell frequencies (darker = higher count) + Annotations showing: P(Compiled)=1600/2000=0.8, P(Compiled|Baseline)=0.75, P(Compiled|Restricciones)=0.85 + Professional heatmap style with white text on colored cells + Add border highlighting the key comparison cells
```

---

### Imagen STATS-1.3: Árbol Bayesiano
- **Ubicación**: `lecturas/Stats/01-Fundamentos-Probabilidad.md:121`
- **Tipo**: Diagrama de árbol
- **Descripción**: Inversión de probabilidades condicionales

```
Probability tree diagram for Bayes' theorem example + Start node splits into: "Kernel Optimized" (5%) and "Kernel Not Optimized" (95%) + From each, branch further: "Detector says Optimized" with paths + Show all 4 end nodes with joint probabilities calculated + Node sizes represent probability magnitudes + Color coding: Green for true positives, Red for false positives + Include annotations showing: P(Detector|Optimized)=0.95, P(Detector|¬Optimized)=0.10 + Path labels showing intermediate calculations + Final box highlighting: P(Optimized|Detector)≈0.333 with emphasis + Academic style, clear hierarchical layout
```

---

### Imagen STATS-1.4: PMF Bar Chart
- **Ubicación**: `lecturas/Stats/01-Fundamentos-Probabilidad.md:149`
- **Tipo**: Gráfico de barras
- **Descripción**: Distribución discreta de kernels válidos

```
Bar chart showing probability mass function + X-axis: "Número de Kernels Válidos" from 0 to 5 + Y-axis: "Probabilidad p(x)" from 0 to 0.4 + Bars at positions: x=0 (height 0.01), x=1 (0.05), x=2 (0.15), x=3 (0.30), x=4 (0.35), x=5 (0.14) + Fill color: teal/steel blue with slight transparency + Border: dark gray + Vertical gridlines at y-axis ticks for readability + Axis labels with units and legend + Add horizontal reference line at E[X]=3.35 with annotation "Expected Value" + Statistics box showing: E[X]=3.35, Var(X)=... + High-quality academic chart
```

---

## 02-Distribuciones-Descriptiva.md

### Imagen STATS-2.1: Comparación de 4 Distribuciones
- **Ubicación**: `lecturas/Stats/02-Distribuciones-Descriptiva.md:97`
- **Tipo**: Overlay density plots
- **Descripción**: Bernoulli, Binomial, Poisson, Normal

```
Four probability distribution curves overlaid on same plot + X-axis: "Valor" from -1 to 10 + Y-axis: "Densidad/Probabilidad" from 0 to 0.4 + Curve 1: Bernoulli(p=0.82) - discrete points at 0,1 + Curve 2: Binomial(n=50, p=0.82) - bell-shaped discrete bars + Curve 3: Poisson(λ=5) - discrete distribution + Curve 4: Normal(μ=5, σ=1.15) - smooth bell curve + Different colors: Red=Bernoulli, Orange=Binomial, Green=Poisson, Blue=Normal + Legend identifying each distribution + Annotation showing "Binomial ≈ Normal when n large" with arrow + Educational quality suitable for statistics textbook
```

---

### Imagen STATS-2.2: Box Plot Media vs Mediana
- **Ubicación**: `lecturas/Stats/02-Distribuciones-Descriptiva.md:235`
- **Tipo**: Box plots comparativos
- **Descripción**: Impacto de outliers en media vs mediana

```
Two box plots side by side + Left plot: Data with outlier [2.1, 2.3, 2.2, 2.0, 15.5] + Right plot: Data without outlier [2.0, 2.1, 2.2, 2.3] + For each: Show box (Q1, median, Q3), whiskers, outlier point (*) + Overlay point markers: red diamond for Mean, blue line for Median + Left: Mean at 4.82 (visibly higher), Median at 2.2 (lower) + Right: Mean and Median much closer + Add annotations: "Mean pulled by outlier" with arrow to 15.5 + Y-axis: "Tiempo de compilación (segundos)" from 0 to 16 + Color scheme: Box=lightblue, Whiskers=black, Points=red + Statistics box showing numeric values + Title: "Impacto de Outliers en Media vs Mediana"
```

---

### Imagen STATS-2.3: Teorema Central del Límite
- **Ubicación**: `lecturas/Stats/02-Distribuciones-Descriptiva.md:135`
- **Tipo**: Three-panel histogram
- **Descripción**: Distribución de medias tiende a normal

```
Three histograms arranged horizontally showing TCL progression + Title: "Teorema Central del Límite: Distribución de Medias Muestrales" + Panel 1 (left): "Distribución Original (Asimétrica)" - Skewed histogram + X-axis: "Valor individual" from 0 to 20 + Y-axis: "Frecuencia" + Bars: Right-skewed distribution (exponential-like) + Panel 2 (middle): "Medias de muestras (n=10)" - Less skewed histogram + X-axis: "Media muestral" + Bars: More symmetric than original + Panel 3 (right): "Medias de muestras (n=100)" - Nearly normal histogram + X-axis: "Media muestral" + Bars: Nearly perfect bell shape + All panels same y-axis scale for comparison + Color progression: Panel 1=red, Panel 2=orange, Panel 3=green + Overlay smooth normal curve on panel 3 to show convergence + Annotation: "n large → Distribution of means → Normal" + Academic quality suitable for statistics education
```

---

### Imagen STATS-2.4: Curva Normal con Z-scores
- **Ubicación**: `lecturas/Stats/02-Distribuciones-Descriptiva.md:117`
- **Tipo**: Distribución normal con áreas
- **Descripción**: Z-score y área de probabilidad

```
Standard normal distribution curve N(0,1) + X-axis: "Z-score" from -4 to 4 with markers at -3,-2,-1,0,1,2,3 + Y-axis: "Densidad" from 0 to 0.45 + Bell curve: Smooth, symmetric, blue fill + Vertical line at z=2.09 in red, labeled "Z = (7.5-5.2)/1.1 = 2.09" + Shade right tail (z > 2.09) in light red with opacity + Add probability label: "P(Z > 2.09) ≈ 0.018" in red box + Mark standard deviations with brackets: μ±σ, μ±2σ, μ±3σ + Add text: "Tiempo de kernel es 2.09 desviaciones estándar arriba de media" + Annotation showing percentage of data within each σ: 68%, 95%, 99.7% + Grid background with light gray lines + Publication-ready quality
```

---

## 03-Pruebas-Hipotesis.md

### Imagen STATS-3.1: Regiones de Rechazo
- **Ubicación**: `lecturas/Stats/03-Pruebas-Hipotesis.md:90`
- **Tipo**: Distribución t con regiones
- **Descripción**: Dónde p < 0.05 rechaza H₀

```
t-distribution curve with rejection regions illustrated + X-axis: "Valor t" from -4 to 4 + Y-axis: "Densidad" from 0 to 0.4 + Main curve: t-distribution (df=24), symmetric bell curve in blue + Left tail region (t < -2.064): Shaded light red, labeled "Rechaza H₀ (α/2=0.025)" + Right tail region (t > 2.064): Shaded light red, labeled "Rechaza H₀ (α/2=0.025)" + Central region (-2.064 to 2.064): Shaded light green, labeled "No rechaza H₀" + Vertical dashed lines at ±2.064 in red + Vertical line at t=3.33 (test statistic from example) in black, bold + Add arrow: "t=3.33 cae en región de rechazo" + Annotation: "p ≈ 0.003 < 0.05 → Rechaza H₀" + Legend and text annotations explaining regions + Title: "Prueba t de una muestra: df=24, α=0.05" + Statistical textbook quality
```

---

### Imagen STATS-3.2: Errores Tipo I y II
- **Ubicación**: `lecturas/Stats/03-Pruebas-Hipotesis.md:39`
- **Tipo**: Matriz 2x2
- **Descripción**: Cuatro resultados posibles

```
2x2 matrix showing Type I and Type II errors + Rows: "Reject H₀" (top), "Fail to Reject H₀" (bottom) + Columns: "H₀ is True" (left), "H₀ is False" (right) + Cell 1 (top-left): "Type I Error (α=0.05)" - Red background, white text + Cell 2 (top-right): "Correct ✓" - Green background + Cell 3 (bottom-left): "Correct ✓" - Green background + Cell 4 (bottom-right): "Type II Error (β)" - Orange background + Add icons: X for errors, checkmark for correct + Text labels in each cell with probability notation + Context example: "Method is actually better but we miss it = Type II" + Add marginal labels: "α = Tasa de falsos positivos", "β = Tasa de falsos negativos" + Annotations: "Power = 1-β" + Professional color scheme matching statistics standards + High contrast for accessibility
```

---

### Imagen STATS-3.3: Tipos de Pruebas t
- **Ubicación**: `lecturas/Stats/03-Pruebas-Hipotesis.md:225`
- **Tipo**: Comparativa de 3 columnas
- **Descripción**: t-una muestra, t-dos muestras, t-pareada

```
Three comparison columns side by side + Column 1: "t de una muestra" + Icon: Single dataset vs reference line + Formula box: "t = (x̄ - μ₀) / (s/√n)" + Example: "¿Tiempo compilación ≠ 5s?" + Color: Blue + Column 2: "t de dos muestras (independientes)" + Icon: Two separate groups + Formula box: "t = (x̄₁ - x̄₂) / SE" + Example: "Baseline vs Restricciones" + Color: Orange + Column 3: "t pareada" + Icon: Same items measured twice (arrows) + Formula box: "t = d̄ / (s_d/√n)" + Example: "Mismo kernel, 2 métodos" + Color: Green + Bottom section: Decision matrix showing when to use each + Sample visualization for each: Small diagrams showing data layout + Assumptions listed under each + Professional educational poster style
```

---

## 04-PowerAnalysis-Diseño.md

### Imagen STATS-4.1: Diseño Experimental
- **Ubicación**: `lecturas/Stats/04-PowerAnalysis-Diseño.md:35`
- **Tipo**: Flowchart
- **Descripción**: IV → manipulation → DV measurement

```
Flowchart showing experimental design process + Flow: "Define IV" → "Assign groups" → "Manipulate/Measure IV" → "Control confounders" → "Measure DV" → "Analyze" → "Conclusions" + Each box contains icons and brief text + Box 1 (Define IV): Show variable icon, example "Baseline vs Restricciones" + Box 2 (Assign): Show two groups splitting from single population + Box 3 (Manipulate): Show intervention arrow + Box 4 (Control): Show checkmarks for controlled variables + Box 5 (Measure): Show measurement tool icon, "Rate de validez" + Box 6 (Analyze): Show statistical test icon, "t-test, p-value" + Box 7 (Conclusions): Show evidence box + Connecting arrows between boxes + Color progression: Green (start) → Yellow (middle) → Red (conclusion) + Decision diamonds at key points + Sidebar showing: "Threats to validity at each stage" + Professional research methodology diagram quality
```

---

### Imagen STATS-4.2: Power Curves
- **Ubicación**: `lecturas/Stats/04-PowerAnalysis-Diseño.md:147`
- **Tipo**: Line plot
- **Descripción**: Poder vs tamaño muestral por effect size

```
Line plot showing power curves for different effect sizes + X-axis: "Tamaño muestral (n)" from 10 to 500 + Y-axis: "Poder estadístico (1-β)" from 0 to 1.0 + Horizontal reference line at power = 0.80 in gray dashed + Multiple curves for different Cohen's d values: + Curve 1 (d=0.8, large effect): Green line, reaches 0.80 power at n≈50 + Curve 2 (d=0.5, medium effect): Orange line, reaches 0.80 at n≈130 + Curve 3 (d=0.2, small effect): Red line, reaches 0.80 at n≈390 + Each curve smoothly increasing, asymptotic to 1.0 + Legend showing color = effect size + Shade region above y=0.80 in light green: "Acceptable power zone" + Annotations on curves showing sample sizes: "n=50 (d=0.8)", "n=130 (d=0.5)", "n=390 (d=0.2)" + Text box: "Pequeños efectos requieren muestras más grandes" + Title: "Curvas de Poder Estadístico: Efecto vs Tamaño Muestral" + Grid background, publication quality
```

---

### Imagen STATS-4.3: Cohen's d Visual
- **Ubicación**: `lecturas/Stats/04-PowerAnalysis-Diseño.md:165`
- **Tipo**: Comparación de distribuciones
- **Descripción**: d = 0.2, 0.5, 0.8 visualmente

```
Four rows showing effect size magnitudes visually + Each row shows two normal curves overlaid (Control vs Treatment) + Row 1: "d = 0.2 (Pequeño)" - Curves almost entirely overlapping, Treatment slightly right-shifted + Row 2: "d = 0.5 (Mediano)" - Curves more separated, clear but partial overlap + Row 3: "d = 0.8 (Grande)" - Curves clearly separated with less overlap + Row 4: "d > 1.2 (Muy Grande)" - Curves mostly non-overlapping + For each row: + Left curve (blue): "Control" centered at 0 + Right curve (red): "Treatment" centered at d value + Vertical dashed lines at means + Shade overlap region in light purple + Label percentage of non-overlap: "54%", "69%", "79%", "88%" + Bottom section: Bar chart showing d values with interpretive labels + All curves same y-axis scale for visual comparison + Color progression making effect sizes visually obvious + Text annotation: "Tamaño del efecto visible: cómo se separan las distribuciones" + Educational quality for understanding practical significance
```

---

## 05-Reproducibilidad.md

### Imagen STATS-5.1: Variabilidad por Semilla
- **Ubicación**: `lecturas/Stats/05-Reproducibilidad.md:54`
- **Tipo**: Line plot con bandas
- **Descripción**: Robustez a través de múltiples semillas

```
Line plot showing metric (validez %) across 5 different random seeds + X-axis: "Semilla aleatoria" with labels [42, 123, 456, 789, 999] + Y-axis: "Tasa de validez (%)" from 70 to 90 + Two lines: Blue for Baseline, Red for Restricciones + Blue line data points: ~75, 74, 76, 75, 76 (mean 75%, small variance) + Red line data points: ~85, 86, 84, 85, 86 (mean 85%, small variance) + Connect points with lines + Shade band around each line showing ±1 SD + Horizontal dashed reference lines at means (75% and 85%) + Add text box: "Conclusión: Efecto robusto a variación de semilla" + Add annotation showing consistent gap between lines: "10% difference maintained" + Top box: "SEED=42: 75%", "SEED=123: 74%", ... showing exact values + Legend: "Baseline", "Restricciones" + Title: "Robustez del Efecto a Variación de Semilla Aleatoria (Temperature=0)" + Professional experimental results visualization
```

---

### Imagen STATS-5.2: Temperatura vs Determinismo
- **Ubicación**: `lecturas/Stats/05-Reproducibilidad.md:79`
- **Tipo**: Heatmap
- **Descripción**: Temperature=0 determinístico vs temperature>0 varía

```
Heatmap showing variability across different temperatures and seeds + X-axis: "Temperatura" [0.0, 0.3, 0.5, 0.7, 1.0] + Y-axis: "Semilla" [42, 123, 456, 789, 999] + Cell values: Coeficiente de variación (CV) o desviación estándar + Cell colors represent variance: + Dark blue (low variance): Temperature=0.0 (all cells ~0.2%) + Light blue (medium): Temperature=0.3 (values ~5%) + Yellow (medium-high): Temperature=0.5 (values ~12%) + Orange (high): Temperature=0.7 (values ~18%) + Red (very high): Temperature=1.0 (values ~25%) + Each cell shows numeric value + Vertical dashed line at temperature=0 with label: "Determinístico" + Add horizontal arrow spanning temperature=0 row: "Mismo resultado cada vez" + Box highlighting temperature=0.0 column: "Reproducibilidad Garantizada" + Bottom explanation: "Temperature=0 → Determinístico (research publication) vs Temperature>0 → Estocástico (realista)" + Title: "Variabilidad de Resultados por Temperatura y Semilla" + Publication-quality heatmap
```

---

## 06-Comparaciones-Multiples.md

### Imagen STATS-6.1: FWER Problem
- **Ubicación**: `lecturas/Stats/06-Comparaciones-Multiples.md:20`
- **Tipo**: Bar chart
- **Descripción**: Error rate crece con número de pruebas

```
Bar chart showing Family-Wise Error Rate as function of number of tests + X-axis: "Número de pruebas independientes" [1, 5, 10, 15, 20, 30, 50] + Y-axis: "Probabilidad de al menos un Error Tipo I (FWER)" from 0 to 1.0 + Red bars showing uncorrected FWER: 0.05, 0.23, 0.41, 0.54, 0.64, 0.78, 0.97 + Blue bars showing Bonferroni-corrected FWER: 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 + Each bar pair grouped together + Red bars dramatically increase, blue bars flat + Horizontal dashed line at FWER=0.05 in green: "Target significance level" + Annotation with arrow pointing to red bars: "Sin corrección: ¡64% de falso positivo con 20 pruebas!" + Annotation pointing to blue bars: "Con Bonferroni: Controlado a 5%" + Legend: "Sin corrección", "Con corrección Bonferroni" + Add summary box: "Más pruebas → Mayor riesgo de hallazgos falsos" + Title: "Tasa de Error Familia-wise (FWER): Impacto de Comparaciones Múltiples" + Statistical education quality
```

---

### Imagen STATS-6.2: Métodos de Corrección
- **Ubicación**: `lecturas/Stats/06-Comparaciones-Multiples.md:61`
- **Tipo**: Line plot
- **Descripción**: Bonferroni vs Holm vs sin corrección

```
Line plot showing p-value thresholds for different correction methods + X-axis: "Número de prueba (ordenada por p-valor ascendente)" from 1 to 20 + Y-axis: "Umbral α ajustado" from 0 to 0.06 + Curve 1: "Sin corrección" - Horizontal line at 0.05 in red, very lenient + Curve 2: "Bonferroni" - Horizontal line at 0.05/20=0.0025 in blue, very conservative + Curve 3: "Holm" - Stepped curve starting at 0.05/20≈0.0025, increasing to 0.05 in green + Curve 3 shows steps upward as i increases + Shade region above each curve in light color matching curve + Horizontal gridlines at key α values: 0.001, 0.01, 0.05 + Legend explaining which method is shown + Annotation: "Holm less conservative than Bonferroni, more power recovered" + Example box: "5th test: p=0.024 → Bonferroni rechaza (p > 0.0025) pero Holm acepta (p < 0.01)" + Title: "Comparación de Métodos de Corrección por Comparaciones Múltiples" + Statistical methodology visualization
```

---

### Imagen STATS-6.3: ANOVA Varianza
- **Ubicación**: `lecturas/Stats/06-Comparaciones-Multiples.md:84`
- **Tipo**: Diagrama de partición
- **Descripción**: Varianza Entre vs Dentro

```
Diagram showing ANOVA variance partitioning concept + Top section: "Varianza Total (SST)" - Large circle/rectangle containing all variation + Branching down: Arrow splitting into two components + Left branch: "Varianza Entre Grupos (SSB)" - Green box, labeled "Difference between group means" + Right branch: "Varianza Dentro de Grupos (SSW)" - Orange box, labeled "Variation within groups" + Visual representation with three groups: + Group A (blue points scattered): mean at 42.8, points 42-45 + Group B (red points scattered): mean at 38.4, points 37-40 + Group C (green points scattered): mean at 36.0, points 34-38 + Larger spread between group means → larger SSB + Smaller scatter within groups → smaller SSW + F ratio = SSB/SSW shown prominently + Bottom section: Formula box showing F = MSB/MSW + Annotation: "Large F → Groups differ" + Color coding helps distinguish between and within variance + Arrows showing calculation steps + Title: "ANOVA: Particionando Varianza Total en Entre y Dentro de Grupos" + Educational statistics diagram quality
```

---

## 07-Pruebas-NoParametricas.md

### Imagen STATS-7.1: Normalidad Test
- **Ubicación**: `lecturas/Stats/07-Pruebas-NoParametricas.md:37`
- **Tipo**: Three-panel histogram
- **Descripción**: Normal vs sesgada con Shapiro-Wilk

```
Three histograms comparing normal vs skewed distributions + Panel 1 (left): "Distribución Normal (p>0.05)" + Histogram: Symmetric bell-shaped distribution + Data: Normal(μ=5, σ=1) simulated + Color: Light blue bars + Overlay: Smooth normal curve in dark blue + Shapiro-Wilk label: "W=0.98, p=0.42 ✓ Normal" in green + Panel 2 (middle): "Distribución Sesgada (p<0.05)" + Histogram: Right-skewed distribution + Data: Iteraciones [42,43,41,45,43,42,88,44,41,43] + Color: Light orange bars + Shapiro-Wilk label: "W=0.89, p=0.001 ✗ No Normal" in red + Annotation: "Outlier at 88 causes skewness" + Panel 3 (right): "Comparación Q-Q Plot" + Scatter plot of theoretical vs observed quantiles + Normal data: Points follow diagonal line closely + Skewed data: Points deviate in S-shape + Diagonal reference line in gray dashed + Color: Points from panel 1 in blue, points from panel 2 in orange + All panels share x-axis "Valor" and y-axis "Frecuencia/Densidad" + Bottom text: "Prueba Shapiro-Wilk detecta desviaciones de normalidad" + Title: "Evaluando Normalidad: Test Shapiro-Wilk" + Statistical diagnostic quality
```

---

### Imagen STATS-7.2: Mann-Whitney U vs t-test
- **Ubicación**: `lecturas/Stats/07-Pruebas-NoParametricas.md:82`
- **Tipo**: Comparación lado a lado
- **Descripción**: Robustez a outliers

```
Two columns comparing parametric vs non-parametric approach + Left column: "t-test (Paramétrico)" + Scatter plot of two groups (Baseline, Restricciones) + Baseline: Points at 4.1, 4.3, 3.9, 4.2, 4.4 + Restricciones: Points at 3.2, 3.1, 3.5, 3.0, 3.3 + Mean lines: Red for Baseline (~4.18), Blue for Restricciones (~3.22) + Normal curves fitted to each group + Result box: "t = 4.51, p = 0.002, d = 2.64 (Muy Grande)" + Right column: "Mann-Whitney U (No paramétrico)" + Same scatter plot but points replaced with rank numbers + Ranks 1-10 assigned, points ordered horizontally + Baseline ranks: 6,7,8,9,10 (sum=40) + Restricciones ranks: 1,2,3,4,5 (sum=15) + Result box: "U = 0, p < 0.001, r = 0.95 (Muy Grande)" + Bottom comparison: Both reach same conclusion (significant difference) + Additional panel showing impact with outlier: + Data with one outlier added to Baseline (value 20) + t-test becomes less significant (p increases) + Mann-Whitney U remains same (rank-based) + Label: "Mann-Whitney U robusto a outliers" + Title: "Comparación: t-test vs Mann-Whitney U (Robustez a Outliers)" + Statistical methods comparison illustration
```

---

## 08-Tamano-Efecto.md

### Imagen STATS-8.1: Cohen's d en GPU
- **Ubicación**: `lecturas/Stats/08-Tamano-Efecto.md:53`
- **Tipo**: Horizontal bar chart
- **Descripción**: Effect sizes en contexto GPU

```
Horizontal bar chart showing Cohen's d values on gradient scale + Y-axis: Escenarios de mejora en GPU + 1. "Compilabilidad: 75% → 85%" → d ≈ 0.30 + 2. "Tiempo: 5.2s → 4.8s" → d ≈ 0.47 + 3. "Iteraciones: 45 → 38" → d ≈ 0.65 + 4. "Throughput: +25% mejora" → d ≈ 0.80 + 5. "Arquitectura nueva" → d ≈ 1.2 + X-axis: "Cohen's d value" from 0 to 1.5 + Each bar colored by magnitude: + Light red (0-0.2): Efecto negligible + Yellow (0.2-0.5): Efecto pequeño + Orange (0.5-0.8): Efecto mediano + Dark green (0.8-1.2): Efecto grande + Dark blue (>1.2): Efecto muy grande + Labels on bars showing exact d value and interpretation + Vertical dashed lines at interpretation boundaries: 0.2, 0.5, 0.8, 1.2 + Legend: "Tamaño del efecto según Cohen (GPU context)" + Bottom reference: "Porcentaje de no-solapamiento: 54%, 69%, 79%, 88%" + Title: "Cohen's d en Contexto: Mejoras Típicas en GPU Kernels" + Practical effect size interpretation visualization
```

---

### Imagen STATS-8.2: Intervalos de Confianza
- **Ubicación**: `lecturas/Stats/08-Tamano-Efecto.md:135`
- **Tipo**: Forest plot
- **Descripción**: Superposición de IC para significancia

```
Forest plot showing 4 comparisons with confidence intervals + Y-axis: Comparación labels ["Baseline", "Método A", "Método B", "Método C"] + X-axis: "Tiempo medio (segundos)" from 4.0 to 5.5 + Four horizontal lines with intervals: + Line 1: Baseline IC [4.51, 5.09], point at 4.8, color blue + Line 2: Método A IC [4.20, 4.80], point at 4.5, color green + Line 3: Método B IC [4.40, 4.70], point at 4.55, color orange + Line 4: Método C IC [4.60, 5.20], point at 4.9, color red + Each interval shown as horizontal line with error bars (endpoints) + Point estimate as diamond or dot in center + Vertical dashed line at overall mean (reference) + Shading to show overlapping intervals in light gray + Annotations: + "Baseline ↔ Método A: No significante (se solapan)" + "Baseline ↔ Método B: Significante (no se solapan)" + Color coding: Green interval = best performance (narrower, leftmost) + Note: "Non-overlapping CIs → Likely significant difference" + Title: "Intervalos de Confianza 95%: Comparación Visual de Métodos" + Statistical inference visualization
```

---

*[El documento continúa con más secciones para Research, Project_1 y Project_2...]*

---

# Módulo Research

> **35 imágenes identificadas** - Ver detalles completos por archivo

Los prompts para este módulo incluyen:
- Diagramas de estructura de tesis
- Timelines metodológicos
- Comparativas de formatos (APA vs IEEE)
- Flowcharts de proceso de investigación
- Matrices de evaluación

---

# Módulo Project_1

> **27 imágenes identificadas** - Ver detalles completos por archivo

Los prompts para este módulo incluyen:
- Arquitecturas LLM
- Pipelines de XGrammar
- Diagramas de gramáticas EBNF
- Comparativas de serving
- Arquitecturas de sistemas agénticos

---

# Módulo Project_2

> **27 imágenes identificadas** - Ver detalles completos por archivo

Los prompts para este módulo incluyen:
- Arquitecturas GPU/CUDA
- Jerarquías de memoria
- Modelos Roofline
- Taxonomías de errores
- Pipelines de benchmark

---

## Instrucciones de Uso

1. **Copiar el prompt** correspondiente a la imagen deseada
2. **Abrir Nano Banana 2** o herramienta de generación preferida
3. **Pegar el prompt** y generar la imagen
4. **Verificar precisión técnica** del contenido generado
5. **Insertar la imagen** en la ubicación indicada del archivo MD
6. **Agregar caption descriptivo** siguiendo formato APA/IEEE

## Nomenclatura Sugerida para Archivos

```
img_{modulo}_{archivo}_{numero}.png

Ejemplos:
- img_ai_01_timeline_evolution.png
- img_comp_02_dfa_hola.png
- img_stats_03_rejection_regions.png
```

---

*Generado: Marzo 2026*
*Total de prompts: ~170*
