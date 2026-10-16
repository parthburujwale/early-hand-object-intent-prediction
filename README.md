# Early Hand-Object Intent Prediction from Partial 3D Interactions

This project investigates how early a human action can be predicted from partial egocentric hand-object interactions.

The central research question is:

> How early can we infer a user's intended action from visual context and partial 3D hand motion?

Using the H2O dataset, we study action prediction at 10%, 20%, 30%, 50%, 75%, and 100% observation of an interaction. The final system combines RGB appearance with 3D hand pose and temporal hand-motion information using hand-conditioned cross-attention.

## Motivation

Early intent prediction is useful for human-robot interaction, assistive systems, anticipatory interfaces, and embodied AI.

A visual model can often recognize which object is present very early in an interaction. However, object identity alone does not reveal what the person intends to do with that object.

For example, seeing a milk container may not distinguish between:

- grabbing milk
- placing milk
- opening milk
- closing milk
- pouring milk

This project investigates whether 3D hand configuration and temporal motion can resolve this ambiguity before the action is completed.

## Dataset

Experiments use the H2O human-object interaction dataset.

The processed dataset contains synchronized:

- egocentric RGB frames
- depth frames
- 3D hand pose
- action annotations

The action space contains 36 valid verb-object combinations.

Examples include:

- grab milk
- place cocoa
- open milk
- close chips
- take out cappuccino
- put in espresso
- apply lotion
- pour milk

The official validation split contains 122 action segments.

## Partial Observation Protocol

Each action is evaluated at:

- 10%
- 20%
- 30%
- 50%
- 75%
- 100%

of the observed action sequence.

The number of real temporal samples increases with observation:

| Observation | Real temporal samples |
|---|---:|
| 10% | 4 |
| 20% | 8 |
| 30% | 12 |
| 50% | 16 |
| 75% | 24 |
| 100% | 32 |

All sequences use a maximum length of 32 with masked padding.

This prevents the model from receiving the same number of temporal observations at every anticipation fraction.

## Models

### RGB Only

A frozen ImageNet-pretrained ViT-B/16 extracts RGB features.

The temporal sequence is processed using a Transformer encoder.

### Hand Motion Only

The hand-only baseline uses wrist-relative 3D hand joint coordinates together with frame-to-frame changes in those coordinates.

This model contains no RGB information.

### RGB + Hand Position Cross-Attention

3D hand-position tokens are used as queries over RGB temporal features using cross-attention.

This tests whether hand-conditioned visual fusion improves prediction independently of explicit hand velocity.

### RGB + Hand Position + Velocity Cross-Attention

The final model combines:

- frozen ViT RGB features
- wrist-relative 3D hand position
- temporal hand-motion features
- hand-conditioned cross-attention
- temporal Transformer fusion

The RGB encoder remains frozen while the multimodal fusion layers are trained.

## Main Results

Results are averaged over three random seeds.

### Top-1 Accuracy

| Observation | Hand Motion Only | RGB Only | RGB + Hand Position | RGB + Hand Position + Velocity |
|---:|---:|---:|---:|---:|
| 10% | 40.98 ± 0.82 | 68.03 ± 2.17 | 72.95 ± 4.34 | **79.51 ± 3.57** |
| 20% | 48.09 ± 3.87 | 71.04 ± 1.71 | 75.68 ± 5.33 | **80.05 ± 2.88** |
| 30% | 51.09 ± 2.50 | 73.22 ± 1.25 | 79.78 ± 1.71 | **81.42 ± 1.71** |
| 50% | 49.73 ± 1.25 | 74.59 ± 1.42 | **81.42 ± 0.95** | **81.42 ± 2.63** |
| 75% | 52.19 ± 2.50 | 74.59 ± 1.42 | 78.69 ± 2.17 | **80.60 ± 1.25** |
| 100% | 54.92 ± 2.46 | 73.22 ± 5.33 | **78.96 ± 4.66** | 76.78 ± 4.04 |

## Early Prediction Result

At only 10% observation:

- RGB only: **68.03%**
- RGB + hand position: **72.95%**
- RGB + hand position + velocity: **79.51%**

This corresponds to:

- +4.92 percentage points from hand-position cross-attention over RGB
- +6.56 percentage points from adding hand motion
- +11.48 percentage points overall over RGB-only

The benefit of explicit hand motion is strongest during early anticipation.

## What Is the Model Learning?

A semantic decomposition of the 10% predictions reveals:

| Model | Action Accuracy | Verb Accuracy | Object Accuracy |
|---|---:|---:|---:|
| RGB only | 68.03% | 70.77% | 94.54% |
| RGB + hand position | 72.95% | 76.78% | 93.17% |
| RGB + hand position + velocity | **79.51%** | **83.06%** | **95.08%** |

The full model improves:

- action recognition by **+11.48 pp**
- verb recognition by **+12.30 pp**
- object recognition by only **+0.55 pp**

This suggests that RGB appearance already provides strong object information, while 3D hand configuration and motion primarily help determine what action is being performed with that object.

## Verb Ambiguity

At 10% observation:

- RGB correct-object / wrong-verb errors: **26.50%**
- Full-model correct-object / wrong-verb errors: **15.57%**

Across all three seeds:

- RGB-only produces 97 same-object / wrong-verb mistakes
- position cross-attention reduces this to 74
- position + velocity reduces it to 57

Thus the full model reduces these semantic errors by approximately 41% relative to RGB-only.

## Motion-Sensitive Actions

Some of the largest improvements from explicit hand motion occur for actions where direction and interaction dynamics are important.

Examples at 10% observation include:

- close milk
- take out cocoa
- open milk
- open chips
- pour milk
- apply spray
- take out cappuccino

For example:

**close milk**

- hand position model: 22.22%
- position + velocity model: 88.89%

The `close milk -> open milk` confusion decreases from 6 errors with position-only fusion to 1 error after adding hand motion.

## Qualitative Examples

Several validation examples consistently demonstrate the benefit of motion across all three random seeds.

Examples include:

| True Action | RGB Prediction | Hand Position Prediction | Motion Model Prediction |
|---|---|---|---|
| close milk | open milk | open milk | **close milk** |
| take out cocoa | put in cocoa | put in cocoa | **take out cocoa** |
| open milk | place milk | place milk | **open milk** |
| take out cappuccino | put in cappuccino | put in cappuccino | **take out cappuccino** |

These examples illustrate that the model often already knows the correct object but needs motion information to infer the correct interaction direction or verb.

## Key Finding

The central result of this project is:

> Visual appearance provides strong early object recognition, while 3D hand position and especially temporal hand motion provide complementary information for resolving verb-level intent ambiguity.

Explicit motion provides its largest benefit during the earliest observation windows, where the action itself has not yet fully unfolded.

## Repository Structure

```text
intent_modeling/
├── src/
│   ├── build_masked_partial_index.py
│   ├── train_rgb_only_vit_masked.py
│   ├── train_hand_motion_only.py
│   ├── train_position_cross_attention_transformer.py
│   ├── train_motion_cross_attention_transformer.py
│   ├── evaluate_final_models.py
│   ├── aggregate_final_results.py
│   ├── plot_final_results.py
│   ├── analyze_class_performance.py
│   ├── analyze_action_semantics.py
│   ├── make_semantic_summary_figures.py
│   ├── analyze_confusion_reduction.py
│   └── make_qualitative_examples.py
│
└── outputs/
    ├── results/
    └── final_evaluation/