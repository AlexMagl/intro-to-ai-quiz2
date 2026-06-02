# Introduction to AI - Quiz #2
**Student:** Aleksandre Maghlakelidze  
**Date:** June 2, 2026  

---

## Assignment: Reverse Engineering a Small Neural Network

**Task:** You are given a 3×3 neural network simulator. The network has 9 input values, 3 hidden neurons, and 2 output neurons. Your task is to investigate the network behavior experimentally.

---

### Task 1 - Conduct Experiments

#### Screenshots of Experiments
*Below are the required 6 screenshots representing key experimental states:*

1. **All Values 0 (Baseline):**
![all0s.png](./screenshots/all0s.png)
2. **Center Cell Active:**
![center.png](./screenshots/center.png)
3. **All Values 1 (Full Saturation):**
![all1s.png](./screenshots/all1s.png)
4. **Corners Active:**
![corners.png](./screenshots/corners.png)
5. **Left Column Active:**
![leftcolumn.png](./screenshots/leftcolumn.png)
6. **Middle Column Active:**
![midcolumn.png](./screenshots/midcolumn.png)

---

#### Experimental Data Table
| # | Experiment Description | Input Matrix (x1-x9) | Hidden Neurons (h1, h2, h3) | Output y1 | Output y2 | Short Observation |
|---|---|---|---|---|---|---|
| 1 | All values are 0 | `[0,0,0, 0,0,0, 0,0,0]` | `[1, 0, 0.5]` | 0.76 | 0.89 | Strong positive bias keeps outputs high even with no input. |
| 2 | All values are 1 | `[1,1,1, 1,1,1, 1,1,1]` | `[1, 2.56, 0.93]` | 0.90 | 0.93 | Max input pushes the network toward its upper limits. |
| 3 | Center cell active | `[0,0,0, 0,1,0, 0,0,0]` | `[-1, 0, 0.89]` | 0.90 | 0.08 | **Critical:** x5 flips h1 to negative, which kills y2. |
| 4 | Corners active | `[1,0,1, 0,0,0, 1,0,1]` | `[1, 2.96, 0.08]` | 0.54 | 0.95 | Corners drive h2 up, which significantly reduces y1. |
| 5 | First row active | `[1,1,1, 0,0,0, 0,0,0]` | `[1, 0.97, 0.71]` | 0.84 | 0.91 | Balanced activation across the top row. |
| 6 | Left column active | `[1,0,0, 1,0,0, 1,0,0]` | `[1, 5.99, 0.15]` | 0.59 | 0.98 | Maxes h2; y1 drops, suggesting h2 is inhibitory to y1. |
| 7 | Middle column active| `[0,1,0, 0,1,0, 0,1,0]` | `[1, 0, 1]` | 0.91 | 0.88 | Strong support for y1; center y2 drop is mitigated by x2/x8. |
| 8 | Right column active | `[0,0,1, 0,0,1, 0,0,1]` | `[1, 0, 0.14]` | 0.57 | 0.90 | Similar to left column; suppresses y1 via h3/h2. |
| 9 | Checkerboard | `[0,1,0, 1,0,1, 0,1,0]` | `[1, 0.05, 0.95]` | 0.90 | 0.88 | High y1 activation; center is 0, so y2 remains high. |
| 10| High contrast | `[1,0,1, 0,1,0, 1,0,1]` | `[1, 2.51, 0.42]` | 0.72 | 0.94 | Mixed pattern; y2 remains high as long as x5 is off. |
| 11| Random pattern | `[0,1,1, 0,0,1, 1,0,0]` | `[1, 0.07, 0.59]` | 0.79 | 0.89 | Typical response for sparse patterns. |
| 12| High Contrast (X) | `[1,0,1, 0,1,0, 1,0,1]` | `[1, 2.51, 0.42]` | 0.72 | 0.94 | Hidden layer averages out the corner/center inputs. |

---

### Questions and Analysis

**1. Which input cells seem most important for output neuron 1?**
The cells in the **middle column ($x_2, x_5, x_8$)** are most important for keeping $y_1$ high. Conversely, the **corner cells ($x_1, x_3, x_7, x_9$)** are most important in a negative way; when they are active, $y_1$ drops significantly (as seen in Experiment #4 and #6).

**2. Which input cells seem most important for output neuron 2?**
The **center cell ($x_5$)** is the most critical cell for $y_2$. While $y_2$ is generally stable around $0.9$, activating $x_5$ alone (Experiment #3) causes $y_2$ to crash to $0.08$. This indicates a strong inhibitory weight from the center cell to the hidden neuron feeding $y_2$.

**3. Which hidden neuron has the strongest influence on each output?**
* **For $y_1$:** **Hidden Neuron 2 ($h_2$)** has the strongest influence. In Experiment #6, when $h_2$ peaks at $5.99$, $y_1$ drops to its lower levels. This implies $h_2$ has a strong negative (inhibitory) weight connection to $y_1$.
* **For $y_2$:** **Hidden Neuron 1 ($h_1$)** has the strongest influence. $y_2$ only experiences a major drop when $h_1$ switches from its resting state of $1$ to $-1$.

**4. Which activation function creates the most noticeable non-linear behavior?**
The hidden layer neurons appear to use a **Sigmoid or Tanh-like** function, but $h_1$ specifically shows a "step" behavior where it snaps between $1$ and $-1$. However, the most noticeable non-linearity is in the **output neurons**, which exhibit "squashing" behavior—despite $h_2$ increasing dramatically, the output $y_1$ stops dropping after a certain point, showing a clear horizontal asymptote.

**5. What happens when very large positive or negative weights are used?**
Large weights lead to **neuron saturation**. When a weight is extremely high, the neuron reaches its maximum (or minimum) activation value very quickly. In the simulator, this is evidenced by values like $5.99$ or $-1.00$, where further increasing the input doesn't change the output, essentially turning the neuron into a binary switch.

**6. What happens when bias is changed?**
The bias determines the **"resting state"** of the network. Because both $y_1$ and $y_2$ are high ($\approx 0.8$) when all inputs are 0, we can conclude the network has a **strong positive bias**. Changing this would shift the threshold required for the inputs to "turn off" or "turn on" the outputs.