import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd

# --- Core Black-Scholes and Implied Volatility Functions ---

def black_scholes(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call and put option.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility of the stock (annual)

    Returns:
    tuple: (call_price, put_price)
    """
    if T == 0 or sigma == 0: # Handle edge cases to avoid division by zero
        call_price = max(0, S - K)
        put_price = max(0, K - S)
        return call_price, put_price

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))

    return call_price, put_price

def vega(S, K, T, r, sigma):
    """Calculates Vega, the sensitivity of option price to volatility."""
    if T == 0 or sigma == 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calculates the implied volatility using the Newton-Raphson method.
    """
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5  # Initial guess

    for i in range(MAX_ITERATIONS):
        price, _ = black_scholes(S, K, T, r, sigma)
        if option_type == 'put':
            _, price = black_scholes(S, K, T, r, sigma)
        
        vega_val = vega(S, K, T, r, sigma)
        
        if vega_val == 0:
            return np.nan # Cannot find solution

        diff = price - market_price
        
        if abs(diff) < PRECISION:
            return sigma
            
        sigma = sigma - diff / vega_val # Newton-Raphson step

    return sigma # Return the last guess if it doesn't converge


# --- Streamlit User Interface ---

st.set_page_config(layout="wide")

st.title("📈 Black-Scholes Option Pricing Model")
st.caption("A simple tool to calculate European option prices and implied volatility.")

# Create two columns for layout
col1, col2 = st.columns([1, 1.5]) 

with col1:
    st.subheader("Model Inputs")
    
    # Use a form to group inputs
    with st.form("inputs_form"):
        S = st.number_input("Current Asset Price (S)", min_value=0.01, value=100.0, step=0.5)
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=0.5)
        T_days = st.number_input("Time to Expiration (in days)", min_value=1, value=30, step=1)
        r_perc = st.number_input("Risk-Free Interest Rate (r %)", min_value=0.0, value=5.0, step=0.1)
        
        # Mode selection: Calculate Price or Implied Volatility
        mode = st.radio("Select Calculator Mode", ('Calculate Option Price', 'Calculate Implied Volatility'), horizontal=True)
        
        sigma_perc = None
        market_price = None
        option_type = None

        if mode == 'Calculate Option Price':
            sigma_perc = st.number_input("Volatility (σ %)", min_value=0.1, value=20.0, step=0.5)
        else:
            market_price = st.number_input("Market Option Price", min_value=0.01, value=5.0, step=0.01)
            option_type = st.radio("Option Type", ('Call', 'Put'), horizontal=True)
            
        submitted = st.form_submit_button("Calculate")

# Convert inputs to required format
T = T_days / 365.0
r = r_perc / 100.0

with col2:
    st.subheader("Calculation Results")
    if submitted:
        if mode == 'Calculate Option Price':
            sigma = sigma_perc / 100.0
            if sigma > 0:
                call_price, put_price = black_scholes(S, K, T, r, sigma)
                st.markdown("#### Option Prices")
                price_col1, price_col2 = st.columns(2)
                price_col1.metric("Call Option Price", f"${call_price:.2f}")
                price_col2.metric("Put Option Price", f"${put_price:.2f}")
            else:
                st.error("Volatility (σ) must be positive.")
        
        elif mode == 'Calculate Implied Volatility':
            iv = implied_volatility(market_price, S, K, T, r, option_type.lower())
            st.markdown("#### Implied Volatility")
            if not np.isnan(iv):
                st.metric(f"Implied Volatility for {option_type}", f"{iv*100:.2f}%")
            else:
                st.error("Could not calculate Implied Volatility. Check inputs or market price.")

    else:
        st.info("Please enter your parameters in the form on the left and click 'Calculate'.")

# --- Explanations and Formulas Expander ---
with st.expander("ℹ️ About the Black-Scholes Model & Formulas"):
    st.markdown("""
    The **Black-Scholes model** is a mathematical model for pricing European-style options. It's based on the idea that one can perfectly hedge the option by buying and selling the underlying asset in a specific way.

    **The five key inputs are:**
    - **Current Asset Price (S):** The price of the underlying asset (e.g., stock).
    - **Strike Price (K):** The price at which the option can be exercised.
    - **Time to Expiration (T):** The time remaining until the option expires, expressed in years.
    - **Risk-Free Interest Rate (r):** The theoretical rate of return of an investment with zero risk.
    - **Volatility (σ):** A measure of the expected fluctuation in the asset's price, expressed as an annualized standard deviation.

    ---
    
    ### Formulas
    The price of a **call option (C)** and a **put option (P)** are given by:

    $C(S, t) = S N(d_1) - K e^{-r(T-t)} N(d_2)$
    
    $P(S, t) = K e^{-r(T-t)} N(-d_2) - S N(-d_1)$

    Where:
    
    $d_1 = \frac{\ln(\frac{S}{K}) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}}$
    
    $d_2 = d_1 - \sigma\sqrt{T-t}$

    And $N(\cdot)$ is the cumulative distribution function (CDF) of the standard normal distribution.
    
    ---
    
    **Implied Volatility (IV)** is the value of σ that makes the Black-Scholes formula price equal to the current market price of the option. It is found using numerical methods as there is no closed-form solution.
    """)