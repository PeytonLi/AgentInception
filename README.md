# Agent Inception

### The Latent-Steered Web Agent Engine

Built for the **Harness Engineering Hackathon 2026**.

**Agent Inception** is a next-generation web automation framework that explores a completely new paradigm for browser agents: replacing traditional, text-heavy DOM prompting with **latent memory steering**.

Inspired by recent breakthroughs in transformer state sharing, Agent Inception transforms dense webpage structures into reusable memory modules that can be dynamically activated during execution. Instead of repeatedly feeding massive HTML payloads into an LLM's context window, the agent navigates using compact prompts augmented by structural memories injected directly into latent space.

The result is a browser agent engineered to remain fast, hyper-efficient, and structurally resilient even as enterprise workflows scale in complexity.

---

## Why We Built This

Today's browser agents suffer from a fundamental bottleneck: they repeatedly convert raw web views into enormous text prompts. As execution workflows grow longer, three critical systemic failure modes emerge:

### 1. The Context Snowball

Every step adds more history. What begins as a compact system prompt quickly balloons into tens of thousands of tokens as past states and repetitive DOM structures accumulate. By step 15 of a typical enterprise workflow, the model spends vastly more compute time re-reading history than solving the task, causing massive latency spikes and compounding infrastructure costs.

### 2. Attention Drift

Models become structurally overwhelmed by irrelevant webpage boilerplate such as navigation menus, sidebars, hidden tracking scripts, and advertisements. When an attention mechanism is forced to parse a 50,000-token prompt to execute a simple button click, query tokens dilute, causing the agent to lose focus on the core objective (*"lost in the middle"* phenomena).

### 3. The Popup Trap

Unexpected asynchronous events—such as cookie consent banners, sudden multi-factor authentication challenges, or promotional modal overlays—instantly disrupt linear text history execution. Recovering from these interruptions traditionally requires heavy prompt engineering and expansive system templates, inflating baseline costs even when no popups are active on the screen.

---

## Our Approach

Agent Inception introduces an optimization concept called **Zero-Prompt Navigation**.

Instead of repeatedly describing webpages using text serialization, the architecture activates pre-computed, text-conditioned memory modules representing structural knowledge about common interfaces and recovery loops. The active, visible chat prompt history remains strictly focused on the user's high-level intent, while structural layout constraints are dynamically loaded into the transformer's internal attention mechanisms.

```plaintext
User Intent
    ↓
Short Prompt
    ↓
Retrieve Relevant Memory Module
    ↓
Inject Structural Guidance
    ↓
Browser Action
    ↓
Adapt to Interruptions (Stealth Steering)
    ↓
Continue Original Task

```

This structural separation allows the agent to:

* **Maintain a flat, compact prompt footprint** across extended workflow Lifetimes.
* **Isolate objective reasoning** from noisy layout changes.
* **Recover from unexpected browser context shifts** instantly.
* **Eliminate exponential token compounding costs.**

---

## Technical Mechanism

Standard causal attention mechanisms map runtime queries against all historical text tokens processed within the visible prompt context buffer:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Agent Inception modifies this formulation without altering the underlying foundational weights of the model. Webpage layouts are compiled offline into highly compressed, text-conditioned Key-Value cache blocks ($K_{\text{bank}}$ and $V_{\text{bank}}$). At runtime, these matrices are appended directly into target hidden layers ($L_{\text{steer}}$) during the forward pass:

$$K_{\text{effective}} = [K_{\text{prompt}} \parallel K_{\text{bank}}]$$

$$V_{\text{effective}} = [V_{\text{prompt}} \parallel V_{\text{bank}}]$$

Where $\parallel$ represents concatenation along the sequence dimension. The visible sequence length stays incredibly small, while the model immediately inherits the spatial and structural awareness required to interface with the DOM.

---

## Stealth Steering

When an unexpected roadblock arises—such as a mandatory 2FA prompt, cookie banner, or a security challenge modal—the execution layer registers a structural mutation anomaly and applies a hot-swap technique called **Stealth Steering**.

```plaintext
[Active Task Context] ──► (Popup Detected) ──► [Hot-Swap: 2FA Steering Block]
                                                        │
[Resume Original Task] ◄── (Popup Cleared) ◄──── [Execute Tool Action]

```

The system momentarily freezes the primary website memory layer, hot-swaps a highly specialized recovery block into the latent space to guide the agent through clearing the obstruction, and then restores the original context configuration. The agent bypasses the popup seamlessly without mutating the user's text history array.

---

## Architecture Overview

The system is organized as an decoupled, event-driven architecture designed to minimize latency between viewport detection and memory injection:

```mermaid
%%{init: {'theme': 'base', 'flowchart': {'scale': 0.75, 'nodeSpacing': 20, 'rankSpacing': 30}}}%%
flowchart LR

A[User Intent] --> B[API Layer - FastAPI]
B --> C[Agent Core]

C --> D[Planner]
C --> E[Memory Router]
C --> F[Executor]

E --> G[Latent Memory Bank]
G --> H[Vector Store / ClickHouse]

F --> I[Browser Engine - Playwright]
I --> J[DOM Mapper]
I --> K[Action Engine]

K --> L[Execution Result]
L --> C

C --> M[Stealth Steering]
M --> N[Interrupt Handler]
N --> I

---

## Demo Narrative

The visual presentation console guides users and judges through three distinct assessment phases to prove the structural validity of latent space steering:

* **Phase 1: The Cost of Traditional Agents**
We showcase a standard text-prompting agent navigating a multi-page checkout or data entry sequence. As it progresses, the console visualizes prompt tokens ballooning out of control, causing observable execution lag and climbing API infrastructure run rates.
* **Phase 2: Activating Agent Inception**
We trigger the same task sequence using our engine. The prompt history metrics line remains flat at near-zero token levels, displaying fast decision iterations and immediate action dispatches.
* **Phase 3: The Chaos Test**
We programmatically introduce a chaotic layout obstruction (an intrusive signup modal). The system instantly catches the disruption, flags it via a visual terminal indicator, deploys the **Stealth Steering** block to clear the overlay, and resumes the core workflow without losing tracking state.

---

## Built With

* **Next.js** — Rich visual telemetry dashboard and real-time agent console view.
* **Python & FastAPI** — Latent-state mapping, vector matching routines, and core inference router.
* **ClickHouse** — Ultra-high-speed column-family data repository for tracking execution logs and indexing memory modules.
* **Playwright** — Headless browser wrapper, mutation monitoring scripts, and direct view port execution.
* **Open-weight LLMs** — Core foundation models modified with low-level attention layer hooks.
* **Memory Inception Concepts** — Structural deep learning frameworks optimizing state cache routing.

---

## Expected Benefits

Optimizing how models reference web structural maps moves automation out of fragile text-wrappers into runtime reality:

* **Minimal Prompt Inflation:** Eliminates the continuous parsing of heavy layout trees.
* **Predictable Attention Focus:** Keeps model reasoning pristine by keeping boilerplate noise out of attention layers.
* **Dynamic Resilience:** Swaps targeted behavior overrides into execution paths dynamically instead of managing sprawling text prompt playbooks.
* **Commercial Scalability:** Lowers processing token dependencies, shifting web automation to flat, commercial software economics.

---

## Looking Ahead

Agent Inception explores a future where automated systems do not need to repeatedly read the digital environments they interact with. By building robust, reusable structural memories that can be hot-swapped into focus at millisecond speeds, we pave the way for true utility-scale AI automation.

We believe the next generation of scalable AI agents will be defined not by the sheer size of their context windows, but by their systemic capacity to decide:

* What to remember
* When to retrieve it
* How to apply it effectively

---

> *"Instead of making AI agents read more, Agent Inception explores whether they can remember better."*

---

## References

* **Harness Engineering Hackathon:** (https://luma.com/harnesshack?tk=Mjg6St)
* **Memory Inception Research Core:** [arXiv:2605.06225](https://arxiv.org/abs/2605.06225)
