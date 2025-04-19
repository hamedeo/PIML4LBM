# PIML4LBM
Machine Learning LBM Simulation

The assignmnet is to develope a machine learning approach for 2DQ9 LBM method using four of the following approches and validate them using Taylor-Green Vortex test and cavity lid driven test.

  (1) Naive BGK model
  
  (2) Symmetric lattice fetching
  
  (3) Mass & momentum conservation
  
  (4) Combined symmetry and conservation




# The method:

     Steps: 1.Data generation 2.Collision Constructor 3. Dataset Building 4. \n
     A function for use to generate data and validate it in four steps \n
     (1) Naive Method -> just satisfy masss conservation continiuity eq. [This model is computationaly expensive and doesn't guarantee physical constraints]\n
     (2) Satisfying symmetric condition byenforcing \\phi_NN be equivarience in respect to D_8 using group averaging [This method doesn't satisfy Postulate 3 mass & moment invarient cond.]\n
     (3) Conservation of both Mass and Monmentum in x and y dir in which we need to use Algebraic fix: [This method does't satisfy Postulate 2 equivariance cond.]\n
     (3.1) Algebraic Reconstruction (Biased for rows 2, 5, & 8). Why?\n
     (3.2) Symmetric Algebraic Reconstruction with group-averaging method\n
     (3.3) Penalize mass and momentum mismatches with a soft constraint in the loss function
     Combined Symmetric-Conservation to satisfy all 4 Postulates at once and be computationally efficient by reducing degrees of freedom for D2Q9 from 90 down to 18
