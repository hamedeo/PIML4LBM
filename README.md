# PIML4LBM
**Physics-Informed Machine Learning for LBM Simulation**

This project aims to develop a machine learning approach for the 2D-Q9 Lattice Boltzmann Method (LBM) using four different modeling strategies. The models are validated using the **Taylor-Green Vortex** and **lid-driven cavity** tests.

---

## 🔧 Methods Implemented

1. **Naive BGK model**  
2. **Symmetric lattice fetching**  
3. **Mass & momentum conservation**  
4. **Combined symmetry and conservation**

---

## 🧠 Approach Overview

The pipeline consists of the following steps:

1. **Data generation**  
2. **Collision operator construction**  
3. **Dataset building**  
4. **Model training and validation**

Each model version reflects a different level of physical constraint:

---

### 1. **Naive Method**
- Enforces only **mass conservation** (continuity equation).
- Computationally expensive and **does not guarantee physical constraints**.
  
### 2. **Symmetric Condition**
- Enforces **D8 symmetry** by averaging over group operations to ensure `φ_NN` is equivariant.
- **Fails to conserve mass & momentum** (violates Postulate 3).

### 3. **Mass and Momentum Conservation**
- Enforces conservation in **x and y directions**.
- **Fails to ensure equivariance** (violates Postulate 2).
  
  - **3.1**: Algebraic Reconstruction (biased for rows 2, 5, 8)  
  - **3.2**: Symmetric Algebraic Reconstruction using group averaging  
  - **3.3**: Soft constraint in the loss function to penalize mass and momentum mismatches

### 4. **Combined Symmetry + Conservation**
- Satisfies **all 4 physical postulates**.
- Reduces degrees of freedom from 90 → **18** for D2Q9, improving efficiency.

---

## 🧪 Tests
- **Taylor-Green Vortex** for flow validation  
- **Lid-driven cavity** to assess physical accuracy

---

Feel free to clone, explore, and adapt the models for more complex LBM simulations or different lattice types.

