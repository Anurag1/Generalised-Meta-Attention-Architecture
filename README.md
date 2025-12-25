# Generalised MetaAttention Architecture (GMAA)

> **A Method-Aware, Epistemic Reasoning Architecture Beyond Transformers**

**Author & Owner:** Anurag Dongare  
**License:** Apache License 2.0  
**Status:** Research / Experimental (Active Development)

---

## 📌 Overview

The **Generalised MetaAttention Architecture (GMAA)** is a novel neural architecture that extends Transformer-based models with **meta-reasoning, epistemic awareness, and self-critique**.

Unlike standard attention mechanisms that operate purely at the token level, **MetaAttention attends over the model’s own reasoning paths**, enabling the system to:

- Distinguish *knowing* from *inferring* and *being unsure*
- Detect incoherence and hallucinations
- Induce symbolic-like rules from neural representations
- Decide when to **answer**, **revise**, or **abstain**

This repository contains a **from-scratch implementation**, training pipelines, and hybrid integration with large language models (LLMs).

---

## 🚀 Key Contributions

✔️ MetaAttention Mechanism  
✔️ Epistemic Confidence Modeling (Know / Infer / Unsure)  
✔️ Self-Critique Loop for Incoherence Detection  
✔️ Rule Induction Head  
✔️ Hybrid LLM + MetaReasoner System  
✔️ Streaming Training on Common Crawl (C4)  

This work introduces a **new class of reasoning-first NLP systems**, not just a larger Transformer.

---

## 🧠 Motivation

Current Large Language Models:

- Optimize fluency, not truth
- Cannot introspect their own uncertainty
- Hallucinate confidently
- Lack epistemic self-control

**GMAA addresses this gap** by embedding *methodological reasoning* directly into the architecture.

> *Transformers learn relations.  
> MetaAttention learns whether those relations deserve belief.*

---

## 🏗️ Architecture

### High-Level Pipeline

