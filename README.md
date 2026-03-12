# chemeng277_final_project
## Machine Learning Prediction of Mechanical Properties for Fiber Reinforced Composites

This project uses a PyTorch-based Neural Network to model the mechanical properties of composite materials as a function of its composite makeup and manufacturing parameters.  This model specifically predicts tensile strength and modulus across three temperatures, -65F, 75F, and 250F. In addition, plots for the sensitivity of these six properties are given in relation to two manufacturing parameters, cure temperature and void %. This information can help inform the manufacturing design space during early stages of composite part development.

### 1. Requirements

Ensure you have the following Python libraries installed:

<pre>
pip install torch pandas numpy matplotlib scikit-learn openpyxl
</pre>

### 2. Download Compiled Data

Downlaod the provided Compiled Data.xlsx file. 

This file contains data compiled from the Composite Materials Handbook-17 (CMH-17) Volume 2. Tensile strength, modulus, and Poisson ratio are provided for common industrial composite materials.

### 3. Predicting Properties of Specific Composite Material Matrix

The file 'chemeng277_final_proj.py' trains and creates function predict_comp() which will predict tensile strength and modulus across three temperatures if provided the following inputs:

* 'Composition':
* 'Type':
* 'Direction':
* 'Fiber':
* 'Matrix':
* 'Processing':
* 'Resin Content':
* 'Max Cure Temp F':
* 'Ply Thickness':
* 'Void Content Max%':

The text inputs must match at least one of the entries in the Compiled Data.xlsx file for 'Composition', 'Type', 'Direction', 'Fiber', 'Matrix', and 'Processing'.

The numerical inputs for 'Resin Content', 'Max Cure Temp F', 'Ply Thickness', and 'Void Content Max%' can be any numerical value.

### Example:

<pre>
  material_prop = {
    'Composition': 'Carbon-Epoxy Prepreg',
    'Type': 'Tape',
    'Direction': 'Unidirectional',
    'Fiber': 'carbon-PAN',
    'Matrix': 'high flow epoxy resin',
    'Processing': 'Autoclave',
    'Resin Content': 0.32,
    'Max Cure Temp F': 350,
    'Ply Thickness': 0.0055,
    'Void Content Max%': 0.05
}

prediction = predict_comp(material_prop)
</pre>

### 4. Sensitivity of Mechanical Properties to Cure and Void %

The file 'chemeng277_final_proj.py' also provides information on how max cure temperature and void % will affect the tensile strength and modulus.

### Void Percentage Sensitivity (0-10%)
- The model simulates a sweep of void percentages to predict tthe effect on tensile properties
- Baseline: Fixed 350F cure temperature

### Cure Temperature Sensitivity (200-450F)
- The model simulates a sweep of max cure temperatures to predict tthe effect on tensile properties
- Baseline: Fixed 0.05% void content

This can be recreated by running Lines 263-334 of 'chemeng277_final_proj.py' which will generate trend plots visualizing the sensitivity of strength/modulus to processing changes.

### Example Plot:
<p align="center">
  <img src="Cure Temp-Strength 75F.png" width="600">
</p>
