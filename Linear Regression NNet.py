# Linear Regression and Neural Network
import csv
import matplotlib.pyplot as plt
import statistics as stats

file_path = 'rgdp-ngdp_correleation.csv'

# Import the CSV file
with open(file_path, 'r') as csv_file:
    csv_dict_reader = csv.DictReader(csv_file)
    dataset = [row for row in csv_dict_reader]

X = [float(row['Year']) for row in dataset]-1958
y = [float(row['Real GDP']) for row in dataset]

# Perform Ordinary Least Squares
def ordinary_least_squares(X, y):

    n = len(X)

    mean_X = sum(X) / n
    mean_y = sum(y) / n

    # Calculate the slope (beta_regressor) and intercept (beta_intercept)
    numerator = sum((X[i] - mean_X) * (y[i] - mean_y) for i in range(n))
    denominator = sum((X[i] - mean_X)**2 for i in range(n))
# Define the regressor and intercept coefficients

    ### beta_regressor: Coefficient for the input feature.
    ### beta_intercept: Coefficient for the intercept.
    
    beta_regressor = numerator / denominator
    beta_intercept = mean_y - beta_regressor * mean_X

    return beta_intercept, beta_regressor

# Call the function to get the coefficients
intercept_coefficient, regressor_coefficient = ordinary_least_squares(X, y)

print("Intercept Coefficient:", intercept_coefficient)
print("Regressor Coefficient:", regressor_coefficient)

n = len(X)
mean_X = sum(X) / n
mean_y = sum(y) / n
numerator = sum((X[i] - mean_X) * (y[i] - mean_y) for i in range(n+1))
denominator = sum((X[i] - mean_X)**2 for i in range(n))
beta_regressor = numerator / denominator
beta_intercept = mean_y - beta_regressor * mean_X
alpha = mean_y - beta_regressor * mean_X

# Calculate the standard deviation of X
std_dev = (sum((X[i] - mean_X)**2 for i in range(n)) / n)**0.5

# Calculate residuals
residuals = [y[i] - (alpha + beta_regressor * X[i]) for i in range(n)]

# Calculate standard errors
se_alpha = (sum(residuals)**2 / ((n - 2) * sum((X[i] - mean_X)**2 for i in range(n))))**0.5 * (1/n + mean_X**2 / sum((X[i] - mean_X)**2 for i in range(n)))**0.5
se_beta = (sum(residuals)**2 / ((n - 2) * sum((X[i] - mean_X)**2 for i in range(n))))**0.5 / sum((X[i] - mean_X)**2 for i in range(n))**0.5

#Calculate t-statistics
t_alpha = alpha / se_alpha
t_beta = beta_regressor / se_beta

# Calculate p-values
p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2))
p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))

print("p-value for alpha:", p_alpha)
print("p-value for beta:", p_beta)

# Calculate R-squared
SST = sum((y[i] - mean_y)**2 for i in range(n))
SSE = sum(residuals)**2
R_squared = 1 - SSE / SST
print("R-squared:", R_squared)

# Calculate Adjusted R-squared
adjusted_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - 2)
print("Adjusted R-squared:", adjusted_R_squared)

# Calculate F-statistic
F = (SST - SSE) / (SSE / (n - 2))
print("F-statistic:", F)

# Calculate p-value for F-statistic
p_F = 1 - stats.f.cdf(F, 1, n - 2)
print("p-value for F-statistic:", p_F)

# Calculate the mean of sum of squares
MSE = SSE / (n - 2)
MSR = (SST - SSE) / 1

# Calculate the standard error of the regression
SER = (SSE / (n - 2))**0.5

# Calculate the t-statistic for the slope
t = beta_regressor / (SSE / (n - 2) / sum((X[i] - mean_X)**2 for i in range(n)))**0.5
#Jarque-Bera test
# Compute skewness and kurtosis
skew = stats.mean((x - mean_X) / std_dev for x in dataset)
kurtosis = stats.mean(((x - mean_X) / std_dev) ** 2 - 3 for x in dataset)

# Compute Jarque-Bera test statistic
n = len(dataset)
jarque_bera = (n / 6) * (skew ** 2 + (1 / 4) * (kurtosis ** 2))

# Jarque-Bera test (you'll need to interpret this manually)
threshold = 5.99  # 95% confidence level for chi-squared distribution with 2 degrees of freedom
is_normal = jarque_bera <= threshold

print("Jarque-Bera test statistic:", jarque_bera)
print("Is normal (at 95% confidence level):", is_normal)

def model_summary(SSE, MSE, MSR, SER, t, jarque_bera, p_JB):
    print("SSE:", SSE)
    print("MSE:", MSE)
    print("MSR:", MSR)
    print("SER:", SER)
    print("t-statistic for slope:", t)
    print("Jarque-Bera test:", jarque_bera) 
    print("p-value for Jarque-Bera test:", p_JB)

# Define the Neural Network class
class NeuralNetwork:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.intercept_coefficient, self.regressor_coefficient = ordinary_least_squares(X, y)

    # Define the feedforward function
    def feedforward(self):
        return self.regressor_coefficient * self.X + self.intercept_coefficient

    # Define the loss function 
    def loss(self):
        return sum((self.feedforward() - self.y)**2)    
    def backpropagation(self):
        learning_rate = 0.01
        n = len(self.X)
        for i in range(1000):
            self.regressor_coefficient -= learning_rate * 2 / n * sum((self.feedforward() - self.y) * self.X)
            self.intercept_coefficient -= learning_rate * 2 / n * sum(self.feedforward() - self.y)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {self.loss()}')
    def gradient_descent(self):
        self.backpropagation()
        return self.regressor_coefficient, self.intercept_coefficient

# Visualize the data and check assumptions of classical linear regression (CLRM)
plt.scatter(['X'], ['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatterplot of Data')
plt.show()

# Check for linearity
plt.scatter(['X'], ['Y'])
plt.plot(['X'], [intercept_coefficient + regressor_coefficient * X[i] for i in range(n)], color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatterplot of Data with Regression Line')
plt.show()

# Check for homoscedasticity
plt.scatter(['X'], residuals)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals vs. X')
plt.show()

# Check for normality
plt.hist(residuals, bins=10, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Check for autocorrelation
plt.scatter(residuals[1:], residuals[:-1])
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Residuals t-1')
plt.ylabel('Residuals t')
plt.title('Residuals t vs. Residuals t-1')
plt.show()

# Check for multicollinearity
plt.scatter(['X'], ['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatterplot of Data')
plt.show()