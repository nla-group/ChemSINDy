import numpy as np
import sympy as sym
import math

class M1:
    name      = "M1"
    M         = 4                                         # Number of species
    N         = math.comb(M+2,2)-1                        # Number of quadratic monomials
    labels    = ['A','P','cat','catA']                    # Species labels
    complexes = ['catA', 'cat*A', 'cat*P']                # Complexes labels
    Q         = len(complexes)
    s         = 4                                         # Number of reactions
    
    def __init__(self, constants):
        self.k1, self.k_1, self.k2, self.k_2 = constants

    def Constants(self):
        self.constants = np.array([self.k1, self.k_1, self.k2, self.k_2])
        return self.constants
        
    def ExactCoeffMatrix(self):
        # Initialize a M x N matrix
        self.C_ex = np.zeros((self.M,self.N))

        # Species indices (on the x vector)
        A, P, cat, catA = 0, 1, 2, 3

        # Column indices for relevant complexes
        catA_col = 3    # catA
        cat_A_col = 7   # cat*A
        cat_P_col = 8   # cat*P

        # Reaction: A + cat → catA (rate k1)
        self.C_ex[A, cat_A_col] -= self.k1
        self.C_ex[cat, cat_A_col] -= self.k1
        self.C_ex[catA, cat_A_col] += self.k1

        # Reaction: catA → A + cat (rate k_1)
        self.C_ex[A, catA_col] += self.k_1
        self.C_ex[cat, catA_col] += self.k_1
        self.C_ex[catA, catA_col] -= self.k_1

        # Reaction: catA → cat + P (rate k2)
        self.C_ex[cat, catA_col] += self.k2
        self.C_ex[P, catA_col] += self.k2
        self.C_ex[catA, catA_col] -= self.k2

        # Reaction: cat + P → catA (rate k_2)
        self.C_ex[cat, cat_P_col] -= self.k_2
        self.C_ex[P, cat_P_col] -= self.k_2
        self.C_ex[catA, cat_P_col] += self.k_2

        return self.C_ex
    
    def ExactKirchhoffMatrix(self):
        # Initialize a Q x Q matrix
        self.K_ex = np.zeros((self.Q,self.Q))

        # Species indices
        catA, Acat, Pcat = 0, 1, 2

        # catA column
        self.K_ex[catA,catA] = -self.k_1 - self.k2
        self.K_ex[Acat,catA] = self.k_1
        self.K_ex[Pcat,catA] = self.k2

        # Acat column
        self.K_ex[catA,Acat] = self.k1
        self.K_ex[Acat,Acat] = -self.k1

        # Pcat column
        self.K_ex[catA,Pcat] = self.k_2
        self.K_ex[Pcat,Pcat] = -self.k_2
        
        return self.K_ex

class M20:
    name      = "M20"
    M         = 6                                         # Number of species
    N         = math.comb(M+2,2)-1                        # Number of quadratic monomials
    labels    = ['A', 'P', 'cat', 
                 'catA','catI', 'catAI']                  # Species labels
    complexes = ['A', 'P', 'cat', 'catA', 
                 'catI', 'catAI', 'cat*A', 
                 'cat*P']                                 # Complexes labels
    Q         = len(complexes)
    s         = 6                                         # Number of reactions
    
    def __init__(self, constants):
        self.k1, self.k_1, self.k2, self.k_2, self.kI, self.kAI = constants

    def Constants(self):
        self.constants = np.array([self.k1, self.k_1, self.k2, self.k_2, self.kI, self.kAI])
        return self.constants
        
    def ExactCoeffMatrix(self):
        # Initialize a M x N matrix
        self.C_ex = np.zeros((self.M,self.N))

        # Species indices
        A, P, cat, catA, catI, catAI = 0, 1, 2, 3, 4, 5

        # Column indices for relevant complexes
        cat_col = 2
        catA_col = 3
        cat_A_col = 9   # cat*A
        cat_P_col = 10   # cat*P

        # Reaction: A + cat → catA (rate k1)
        self.C_ex[A, cat_A_col] -= self.k1
        self.C_ex[cat, cat_A_col] -= self.k1
        self.C_ex[catA, cat_A_col] += self.k1

        # Reaction: catA → A + cat (rate k_1)
        self.C_ex[A, catA_col] += self.k_1
        self.C_ex[cat, catA_col] += self.k_1
        self.C_ex[catA, catA_col] -= self.k_1

        # Reaction: catA → cat + P (rate k2)
        self.C_ex[cat, catA_col] += self.k2
        self.C_ex[P, catA_col] += self.k2
        self.C_ex[catA, catA_col] -= self.k2

        # Reaction: cat + P → catA (rate k_2)
        self.C_ex[cat, cat_P_col] -= self.k_2
        self.C_ex[P, cat_P_col] -= self.k_2
        self.C_ex[catA, cat_P_col] += self.k_2

        # Reaction: cat → catI (rate kI)
        self.C_ex[cat, cat_col] -= self.kI
        self.C_ex[catI, cat_col] += self.kI

        # Reaction: catA → catAI (rate kAI)
        self.C_ex[catA, catA_col] -= self.kAI
        self.C_ex[catAI, catA_col] += self.kAI

        return self.C_ex

    def ExactKirchhoffMatrix(self):
        # Initialize a Q x Q matrix
        self.K_ex = np.zeros((self.Q,self.Q))

        # Species indices
        # ['A', 'P', 'cat', 'catA', 'catI', 'catAI', 'cat*A', 'cat*P']
        A, P, cat, catA, catI, catAI = 0, 1, 2, 3, 4, 5
        Acat, Pcat                   = 6,7

        # catA column
        self.K_ex[catA,catA]  = -self.k_1 - self.k2 - self.kAI
        self.K_ex[Acat,catA]  = self.k_1
        self.K_ex[Pcat,catA]  = self.k2
        self.K_ex[catAI,catA] = self.kAI

        # Acat column
        self.K_ex[catA,Acat] = self.k1
        self.K_ex[Acat,Acat] = -self.k1

        # Pcat column
        self.K_ex[catA,Pcat] = self.k_2
        self.K_ex[Pcat,Pcat] = -self.k_2

        # cat column
        self.K_ex[cat,cat] = -self.kI
        self.K_ex[catI,cat] = self.kI
        
        return self.K_ex

class Szederkenyi:
    name      = "Szederkenyi"
    M         = 6                                                     # Number of species
    N         = math.comb(M+2,2)-1                                    # Number of quadratic monomials
    labels    = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']                  # Species labels
    complexes = [  'x0' ,   'x1' ,   'x2' ,   'x3' ,   'x4' , 'x5',
                 'x1*x1', 'x3*x2', 'x5*x1', 'x5*x3', 'x5*x4']         # Complexes labels
    Q         = len(complexes)
    s         = 9                                                     # Number of reactions
    
    def __init__(self, constants):
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7, self.k8, self.k9 = constants

    def Constants(self):
        self.constants = np.array([self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7, self.k8, self.k9])
        return self.constants
        
    def ExactCoeffMatrix(self):
        # Initialize a M x N matrix
        self.C_ex = np.zeros((self.M,self.N))

        # Species indices (on the x vector)
        x0, x1, x2 = 0, 1, 2
        x3, x4, x5 = 3, 4, 5

        # Column indices for relevant complexes
        x0_col   =  0
        x1_col   =  1
        x2_col   =  2
        x3_col   =  3
        x4_col   =  4
        x5_col   =  5
        x1x1_col =  8
        x3x2_col =  14
        x5x1_col =  22
        x5x3_col =  24
        x5x4_col =  25
        
        # Reaction: x1 + x1 → x2 (rate k1)
        self.C_ex[x1, x1x1_col] -= 2*self.k1
        self.C_ex[x2, x1x1_col] += self.k1

        # Reaction: x2 → x1 + x1 (rate k2)
        self.C_ex[x2, x2_col]   -= self.k2
        self.C_ex[x1, x2_col] += 2*self.k2

        # Reaction: x2 + x3 → x4 (rate k3)
        self.C_ex[x2, x3x2_col] -= self.k3
        self.C_ex[x3, x3x2_col] -= self.k3
        self.C_ex[x4, x3x2_col] += self.k3
        
        # Reaction: x4 → x2 + x3 (rate k4)
        self.C_ex[x4, x4_col] -= self.k4
        self.C_ex[x2, x4_col] += self.k4
        self.C_ex[x3, x4_col] += self.k4
        
        # Reaction: x4 → x4 + x5 (rate k5)
        self.C_ex[x5, x4_col] += self.k5
        
        # Reaction: x3 → x3 + x5 (rate k6)
        self.C_ex[x5, x3_col] += self.k6
        
        # Reaction: x5 → x0      (rate k7)
        self.C_ex[x0, x5_col] += self.k7
        self.C_ex[x5, x5_col] -= self.k7
        
        # Reaction: x1 → x0      (rate k8)
        self.C_ex[x0, x1_col] += self.k8
        self.C_ex[x1, x1_col] -= self.k8
        
        # Reaction: x5 → x1 + x5 (rate k9)
        self.C_ex[x1, x5_col] += self.k9

        return self.C_ex

    def ExactKirchhoffMatrix(self):
        # Initialize a Q x Q matrix
        # [  'x0' ,   'x1' ,   'x2' ,   'x3' ,   'x4' , 'x5'
        #  'x1*x1', 'x3*x2', 'x5*x1', 'x5*x3', 'x5*x4'] 
        self.K_ex = np.zeros((self.Q,self.Q))

        # Complex indices
        x0, x1, x2 = 0, 1, 2
        x3, x4, x5 = 3, 4, 5
        x1x1 =  6
        x3x2 =  7
        x5x1 =  8
        x5x3 =  9
        x5x4 =  10
        
        # x1 column
        self.K_ex[x0, x1]  =   self.k8
        self.K_ex[x1, x1]  = - self.k8

        # x2 column
        self.K_ex[x2  , x2] = - self.k2
        self.K_ex[x1x1, x2] =   self.k2

        # x3 column
        self.K_ex[x3  , x3] = - self.k6
        self.K_ex[x5x3, x3] =   self.k6

        # x4 column
        self.K_ex[x4  , x4] = - self.k4 - self.k5
        self.K_ex[x3x2, x4] =   self.k4
        self.K_ex[x5x4, x4] =   self.k5
        
        # x5 column
        self.K_ex[x0  , x5]  =   self.k7
        self.K_ex[x5  , x5]  = - self.k7 - self.k9
        self.K_ex[x5x1, x5]  =   self.k9

        # x1*x1 column
        self.K_ex[x2  , x1x1] = self.k1
        self.K_ex[x1x1, x1x1] = -self.k1

        # x3*x2 column
        self.K_ex[x4  , x3x2] = self.k3
        self.K_ex[x3x2, x3x2] = -self.k3
        
        return self.K_ex

class VdV:
    name      = "Van de Vusse"
    M         = 4                                         # Number of species
    N         = math.comb(M+2,2)-1                        # Number of quadratic monomials
    labels    = ['x1','x2','x3','x4']                     # Species labels
    complexes = ['x1','x2','x3','x4', 'x1*x1']            # Complexes labels
    Q         = len(complexes)
    s         = 3                                         # Number of reactions
    
    def __init__(self, constants):
        self.k1, self.k2, self.k3 = constants

    def Constants(self):
        self.constants = np.array([self.k1, self.k2, self.k3])
        return self.constants
        
    def ExactCoeffMatrix(self):
        # Initialize a M x N matrix
        self.C_ex = np.zeros((self.M,self.N))

        # Species indices (on the x vector)
        x1, x2, x3, x4 = 0, 1, 2, 3

        # Column indices for relevant complexes
        x1_col   = 0
        x2_col   = 1
        x3_col   = 2
        x4_col   = 3
        x1x1_col = 4

        # Reaction: x1 + x1 → x2 (rate k1)
        self.C_ex[x1, x1x1_col] -= 2*self.k1
        self.C_ex[x2, x1x1_col] += self.k1
        
        # Reaction: x1 → x3 (rate k2)
        self.C_ex[x1, x1_col] -= self.k2
        self.C_ex[x3, x1_col] += self.k2

        # Reaction: x3 → x4 (rate k3)
        self.C_ex[x3, x3_col] -= self.k3
        self.C_ex[x4, x3_col] += self.k3

        return self.C_ex

    def ExactKirchhoffMatrix(self):
        # Initialize a Q x Q matrix
        # ['x1','x2','x3','x4', 'x1*x1']
        self.K_ex = np.zeros((self.Q,self.Q))

        # Complex indices
        x1   = 0
        x2   = 1
        x3   = 2
        x4   = 3
        x1x1 = 4
        
        # x1 column
        self.K_ex[x1, x1] = - self.k2
        self.K_ex[x3, x1] = self.k2

        # x2 column
        self.K_ex[x2, x2] = 0

        # x3 column
        self.K_ex[x3, x3] = - self.k3
        self.K_ex[x4, x3] = self.k3

        # x4 column
        self.K_ex[x4, x4] = 0
        
        # x1*x1 column
        self.K_ex[x2  , x1x1] = self.k1
        self.K_ex[x1x1, x1x1] = -self.k1

        return self.K_ex



