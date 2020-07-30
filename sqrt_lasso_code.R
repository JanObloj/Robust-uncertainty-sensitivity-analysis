#Author: Johannes Wiesel with modifications by Jan Obloj
#Version 30/07/2020

library("glmnet")
library(matlib)

# Sqrt-Lasso implementation from RPtests: 
# Goodness of Fit Tests for High-Dimensional Linear Regression Models
# https://rdrr.io/cran/RPtests/src/R/sqrt_lasso.R

sqrt_lasso <- function(x, y, lam0=NULL, exclude = integer(0), output_all = FALSE, ...) {
  # Do this first so error checking can take place
  out <- glmnet(x, y, exclude=exclude, intercept = FALSE, ...)
  
  n <- nrow(x)
  p <- ncol(x) - length(exclude)
  
  resids <- y - predict.glmnet(out, newx=x)
  full_MSE <- colMeans(resids^2)
  index_sel <- which.min(abs((full_MSE)*lam0^2 / out$lambda^2 - 1))
  if (index_sel == length(full_MSE)) warning("Smallest lambda chosen by sqrt_lasso")
  if (output_all) {
    return(list("beta"=as.numeric(out$beta[, index_sel]),
                "a0"=out$a0[index_sel],
                "sigma_hat"=sqrt(full_MSE[index_sel]),
                "glmnet_obj"=out))
  }
  return(as.numeric(out$beta[, index_sel]))
}

set.seed(8675309)
###################################
# Input parameters for linear model
N <- 2000
n <- 10
x <- matrix(rnorm(N*n), N, n)
z <- c(rnorm(n))
z <- round(z,2)
## Below is a fixed version of the regression used to produce the plot in the paper. 
y <- 1.5*x[, 1] - 3*x[, 2] - 2*x[,3] +0.3*x[,4]- 0.5*x[,5]-0.7*x[,6]+0.2*x[,7]+0.5*x[,8]+1.2*x[,9]+0.8*x[,10] + 2*rnorm(N)
## A randomized version which generates a different regression each time is also available.
# y <- (x %*% z)[,1] + sqrt(n)*rnorm(N)

###################################
# Fit linear model (without intercept)
linearMod <- lm(y ~0 + x) 
ols_coefficients <- matrix(linearMod$coefficients)
summary(linearMod)

# Calculate theoretical shrinkage
V <- sqrt(sum(resid(linearMod)^2)/N)
D <- 1/N* t(x) %*% x
D_inv <- inv(D)

# Set up Lasso, ridge, sqrt-lasso regression
# lambda <- seq(0.01, 0.2, by=0.04)
lambda <- seq(0.01, 0.1, by=0.02)
sqrt_lasso_coefficients <-  matrix(0, length(lambda),dim(x)[2])
lasso_coefficients <-  matrix(0, length(lambda),dim(x)[2])
ridge_coefficients <-  matrix(0, length(lambda),dim(x)[2])
shrinkage <-  matrix(0, length(lambda),dim(x)[2])
shrinkage_FO <-  matrix(0, length(lambda),dim(x)[2])


#Ridge, Lasso and Sqrt-Lasso regression (without intercept)
for (i in 1:length(lambda)){
  #Ridge
  ridge_reg <- glmnet(x, y, alpha = 0, intercept = FALSE,
                    family = 'gaussian', lambda = lambda[i])
  ridge_coefficients[i,] <- as.numeric(ridge_reg$beta)
  
  #Lasso
  lasso_reg <- glmnet(x, y, alpha = 1, intercept = FALSE,
                    family = 'gaussian', lambda = lambda[i])
  lasso_coefficients[i,] <- as.numeric(lasso_reg$beta)
  
  #Sqrt-Lasso  
  sqrt_lasso_coefficients[i, ] <- sqrt_lasso(x, y, lam0=lambda[i])
  
  #Calculate shrinkage (actual and theoretical)
  shrinkage[i,] <- linearMod$coefficients - sqrt_lasso_coefficients[i,]
  shrinkage_FO[i,] <- (D_inv %*% sign(ols_coefficients) * V *lambda[i] )
}

# Shrinkage plot (actual and theoretical)
par(mar=c(6.1, 4.1, 4.1, 10.1), xpd=TRUE)
plot(shrinkage[1,], type='p', xlab="covariate's index", ylab="shrinkage", main="Parameter Shirnkage: Exact (o) vs First Order Approximation (x)" ,
			ylim=c(min(min(shrinkage),min(shrinkage_FO)),
            max(max(shrinkage),max(shrinkage_FO))) )
legend_text <- c()
for (i in 1:length(lambda)){
  #lines(shrinkage[i,],lty=1, col=i, lwd=2)
  #lines(shrinkage_theoretical[i,], lty=3, col=i, lwd=2)
  points(shrinkage[i,],pch=1,col=i)
  points(shrinkage_FO[i,], pch=4, col=i)
  legend_text <- c(legend_text, paste(expression(delta), "=", lambda[i]))
  
}

legend('bottomright', legend = legend_text, lty=rep(1,length(lambda)), col=1:length(lambda))

