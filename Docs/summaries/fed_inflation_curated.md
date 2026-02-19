# The Hidden Heterogeneity of Inflation Expectations and its Implications
## Curated Technical Extraction

Source: Drager, Lamla, Pfajfar (2020). FEDS 2020-054, Federal Reserve Board.
JEL: E31, E52, E58, D84

---

## 1. CORE ECONOMETRIC FRAMEWORK

### 1.1 Extended Euler Equation Model (Primary Specification)

The central estimation model augments the standard consumption Euler equation with opinion dummies:

```
c^dur_i = a_0 + b_1 * c^{dur,e}_i + b_2 * r^e_{savings,i} + c' * X^opinions_i + d' * X^controls_i + u_i
```

Where:
- `c^dur_i` = log current durable goods spending of household i (in EUR, previous month)
- `c^{dur,e}_i` = expected/planned durable goods spending (qualitative: more/same/less, next 12 months)
- `r^e_{savings,i}` = subjective perceived real interest rate on savings
- `X^opinions_i` = vector of opinion dummies on inflation and interest rates
- `X^controls_i` = vector of demographic controls
- `u_i` = error term

**Theoretical sign predictions**: b_1 > 0 (positive relation between planned and current spending), b_2 < 0 (higher real rate => postpone spending)

### 1.2 Perceived Real Interest Rate Construction

```
r^e_{savings} = i^e_{savings} - pi^e
```

Where:
- `i^e_{savings}` = expected nominal savings interest rate (12-month ahead point forecast)
- `pi^e` = expected inflation rate (12-month ahead point forecast)

This is a Fisher equation decomposition applied at the individual household level using subjective expectations.

### 1.3 Extended Models with Interaction Terms

Full specification with opinion-real-rate interactions:

```
c^dur_i = a_0 + b_1 * c^{dur,e}_i + b_2 * r^e_{savings,i}
         + c_1 * d_inf_lowbetter_i + c_2 * d_inf_highbetter_i
         + c_3 * d_int_lowbetter_i + c_4 * d_int_highbetter_i
         + g_1 * (r^e_{savings,i} * d_inf_lowbetter_i)
         + g_2 * (r^e_{savings,i} * d_inf_highbetter_i)
         + g_3 * (r^e_{savings,i} * d_int_lowbetter_i)
         + g_4 * (r^e_{savings,i} * d_int_highbetter_i)
         + d' * X^controls_i + u_i
```

This nests multiple testable hypotheses:
- g_3 < 0 implies consumers preferring lower interest rates have higher real-rate elasticity
- Reference group: those who consider inflation/interest rates "appropriate"

---

## 2. ESTIMATION METHODS

### 2.1 Probit Models (Opinion Determinants)

- **Method**: Weighted probit models with population weights
- **Output reported**: Average marginal effects evaluated at sample mean
- **Dependent variables**: Binary opinion indicators {d_inf_lowbetter, d_inf_reason, d_inf_highbetter} and {d_int_lowbetter, d_int_reason, d_int_highbetter}
- **Standard errors**: Robust (heteroskedasticity-consistent)
- **Significance levels**: *** p<0.01, ** p<0.05, * p<0.1
- **Fit statistics**: Chi-squared test statistic, Pseudo R-squared

### 2.2 OLS Models (Spending Levels)

- **Method**: Weighted OLS with population weights
- **Dependent variable**: log(spending in EUR) for previous month
- **Truncation**: Highest 5% of spending values excluded
- **Standard errors**: Robust
- **Fit statistic**: Adjusted R-squared

### 2.3 Probit Models (Planned Spending)

- **Method**: Weighted probit with population weights
- **Dependent variable**: Binary indicator for planning to spend MORE in next 12 months
- **Output**: Average marginal effects
- **Note on interactions**: In non-linear probit, interaction effects cannot be interpreted directly from coefficients; graphical analysis required (see appendix figures A.1-A.4)

### 2.4 Lowess Smoothing (Visualization)

Locally Weighted Scatterplot Smoothing (LOWESS) applied to individual observations to produce smoothed conditional share plots of opinion categories against levels of expectations.

---

## 3. VARIABLE DEFINITIONS AND DATA CONSTRUCTION

### 3.1 Opinion Variables (Categorical -> Dummy)

**Inflation opinions** (Wave 1):
- `d_inf_highbetter` = 1 if "Higher inflation than expected would be better"
- `d_inf_reason` = 1 if "Inflation will be more or less appropriate" (REFERENCE CATEGORY)
- `d_inf_lowbetter` = 1 if "Lower inflation than expected would be better"

**Interest rate opinions** (Wave 2):
- `d_int_highbetter` = 1 if "Higher interest rate than expected would be better"
- `d_int_reason` = 1 if "Interest rate will be more or less appropriate" (REFERENCE CATEGORY)
- `d_int_lowbetter` = 1 if "Lower interest rate than expected would be better"

### 3.2 Expectations Variables (Quantitative Point Forecasts, 12-month ahead)

- `pi^e` = expected consumer price inflation rate
  - Truncated to range [-5%, +25%]
- `i^e_{savings}` = expected average savings interest rate
  - Truncated to <= 25%
- `i^e_{mortgage}` = expected average mortgage rate
  - Truncated to <= 25%

### 3.3 Spending/Saving Variables

**Current (levels in EUR, previous month, log-transformed, top 5% truncated)**:
- `c_dur` = durable goods
- `c_cons` = consumption goods
- `c_house` = housing (rent or mortgage payments)
- `saving` = financial reserves

**Planned (qualitative, next 12 months -> binary dummies for "more")**:
- `c_{dur,e}` = plan to spend more on durables
- `c_{cons,e}` = plan to spend more on consumption
- `c_{house,e}` = plan to spend more on housing
- `saving_e` = plan to save more

### 3.4 Demographic Controls

| Variable | Definition |
|----------|-----------|
| `d_male` | Binary: male = 1 |
| `age` | Continuous |
| `inc_low` | Monthly net income <= 1,000 EUR |
| `inc_middle` | Monthly net income 1,000-3,000 EUR |
| `inc_high` | Monthly net income > 3,000 EUR |
| `edu_haupt` | Hauptschule (lowest secondary) |
| `edu_real` | Realschule (medium secondary) |
| `edu_abi` | Abitur (university-qualifying secondary) |
| `edu_uni` | University degree |
| `d_fulltime` | Full-time employment |
| `d_parttime` | Part-time employment |
| `d_noemploy` | Not employed |
| `d_retired` | Retired |
| `d_ownhouse` | Homeowner |
| `d_renthouse` | Renter |
| `d_east1989` | Lived in GDR (East Germany) before 1989 |

---

## 4. SAMPLE STRATIFICATION STRATEGY

All models are estimated across four sample partitions by inflation expectation range:

| Partition | Range | Label |
|-----------|-------|-------|
| Full sample | -5% <= pi^e <= 25% | Unrestricted |
| Low/deflation | -5% <= pi^e < 1.5% | Below target |
| Target zone | 1.5% <= pi^e <= 2% | ECB-anchored |
| High | 2% < pi^e <= 25% | Above target |

This stratification is the core methodological innovation: testing whether demographic and opinion effects differ across expectation regimes.

---

## 5. DATA STRUCTURE

### 5.1 Survey Design

- **Source**: Bundesbank Online Pilot Survey on Consumer Expectations
- **Population**: Representative sample of German households
- **Fielding**: April-June 2019, three waves
- **Total observations**: 6,653
  - Wave 1: N = 2,009
  - Wave 2: N = 2,052
  - Wave 3: N = 2,592
- **Panel component**: ~500 respondents per wave-pair overlap
  - ~500 in waves 1 & 2
  - ~500 in waves 2 & 3
  - ~500 in waves 1 & 3
  - ~500 in all three waves
- **Analysis sample**: ~1,000 participants with responses in waves 1 and 2

### 5.2 Wave-Question Mapping

- Wave 1: Inflation opinion questions + inflation point forecasts
- Wave 2: Interest rate opinion questions + interest rate point forecasts + spending/saving data
- Cross-wave matching: Wave 1 inflation responses linked to Wave 2 interest rate and spending data

---

## 6. KEY QUANTITATIVE FINDINGS

### 6.1 Cross-Tabulation of Opinions (Table 1)

Joint distribution of inflation and interest rate opinions (% of sample):

|  | int_highbetter | int_reason | int_lowbetter | Total |
|--|---------------|------------|---------------|-------|
| **inf_highbetter** | 3.4 | 1.2 | 0.2 | 4.8 |
| **inf_reason** | 28.9 | 7.4 | 1.9 | 38.3 |
| **inf_lowbetter** | 43.0 | 8.1 | 5.8 | 56.9 |
| **Total** | 75.3 | 16.8 | 7.9 | 100.0 |

Key statistics:
- 56.9% believe inflation should be lower
- 75.3% believe interest rates should be higher
- Only 7.4% believe both inflation AND interest rates are appropriate
- 43.0% want lower inflation AND higher interest rates (consistent with Taylor rule)

### 6.2 Hidden Heterogeneity in Target Zone (pi^e in [1.5%, 2%])

Among consumers with ECB-target-consistent expectations:
- 49% believe expected inflation is appropriate
- 46% believe expected inflation should be lower
- 5% believe expected inflation should be higher

Among consumers expecting deflation: ~30% still prefer LOWER inflation.

### 6.3 Probit Model Results -- Inflation Opinions (Table 2)

**Full sample marginal effects on P(inf_lowbetter):**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| d_male | -0.084 | (0.027) | *** |
| inc_high | -0.143 | (0.082) | * |
| d_edu_uni | -0.197 | (0.039) | *** |
| d_east1989 | +0.132 | (0.036) | *** |
| d_ownhouse | -0.046 | (0.028) | * |
| d_noemploy | +0.121 | (0.052) | ** |

**Target zone [1.5, 2] marginal effects on P(inf_lowbetter):**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| d_male | -0.083 | (0.042) | ** |
| inc_high | -0.323 | (0.140) | ** |
| d_edu_uni | -0.289 | (0.057) | *** |
| d_east1989 | +0.198 | (0.051) | *** |
| inc_middle | -0.256 | (0.139) | * |
| d_edu_abi | -0.153 | (0.068) | ** |

Critical finding: Marginal effects are LARGER in the target zone than in the full sample, indicating most demographic heterogeneity in opinions occurs precisely where expectations appear "anchored."

**Full sample N**: 1,515 | Pseudo R-squared: 0.055 (lowbetter), 0.047 (reason), 0.054 (highbetter)

### 6.4 Probit Model Results -- Interest Rate Opinions (Table 3)

**Full sample marginal effects on P(int_highbetter):**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| age | +0.003 | (0.001) | ** |
| inc_high | +0.122 | (0.073) | * |
| d_edu_abi | +0.168 | (0.044) | *** |
| d_edu_uni | +0.090 | (0.037) | ** |

**Full sample N**: 1,616 | Pseudo R-squared: 0.087 (lowbetter), 0.019 (reason), 0.026 (highbetter)

### 6.5 Euler Equation Results -- Durable Spending (Table 4)

**Current durable spending (OLS on log EUR):**

| | Full Sample (1) | Full Sample (2) | Target Zone | Target Zone + Interactions |
|--|----------------|-----------------|-------------|---------------------------|
| c^{dur,e} | 0.166 | 0.123 | 0.212** | 0.060 |
| r^e_{savings} | -0.014 | -0.012 | -0.036* | -0.047*** |
| d_int_lowbetter | -- | 0.681** | 1.045** | 0.250 |
| r^e_{sav} * d_int_lowbetter | -- | -- | -- | -0.986*** |
| N | 379 | 380 | 163 | 163 |
| Adj. R-squared | 0.022 | 0.041 | 0.079 | 0.125 |

Key coefficient: r^e_{sav} * d_int_lowbetter = -0.986 (SE = 0.279, p < 0.01) in target zone. Consumers preferring lower interest rates have nearly 1-to-1 real rate elasticity of durable spending -- dramatically higher than the baseline effect.

Adding opinions increases Adj. R-squared from 0.022 to 0.041 (86% increase in explanatory power).

**Planned durable spending (Probit, marginal effects on P(spend more)):**

| | Full Sample | Target Zone |
|--|------------|-------------|
| d_int_lowbetter | -0.137** / -0.273*** | -0.311** / -0.596*** |

Sign reversal confirms intertemporal substitution: those preferring lower rates spend MORE today and plan to spend LESS tomorrow.

### 6.6 Savings Results (Table 5)

**Current saving (OLS on log EUR):**

| | Full Sample | Target Zone |
|--|------------|-------------|
| saving_e | 0.226*** | 0.330*** |
| d_inf_lowbetter | -0.282*** | -0.197* (becomes insignificant in restricted sample) |
| N | 609 | 264 |
| Adj. R-squared | 0.145 | 0.236 |

Consumers who think inflation should be lower save LESS -- consistent with pessimistic consumers having non-anchored expectations and lower savings.

### 6.7 Homeowner vs. Renter Heterogeneity (Table 6)

**Current durable spending with interaction terms (OLS, full sample):**

| Variable | Homeowners | Renters |
|----------|-----------|---------|
| r^e_{savings} | -0.150* | 0.517 (insig) |
| d_inf_lowbetter | -0.343 (insig) | 0.804 (insig) |
| d_inf_lowbetter (no interactions) | -0.446*** | +0.674** |
| d_int_lowbetter (no interactions) | -0.270 (insig) | +1.043*** |
| r^e_{sav} * d_int_lowbetter | -0.180** | -0.471 (insig) |
| r^e_{sav} * d_int_highbetter | +0.155* | -0.643 (insig) |
| N | 243 | 115 |
| Adj. R-squared | 0.095 | 0.130 |

Asymmetric findings:
- Homeowners who prefer lower inflation spend LESS on durables (consistent with standard Euler equation)
- Renters who prefer lower inflation spend MORE on durables (opposite sign)
- Homeowners show significant real-rate sensitivity; renters do not
- Homeowners' durable spending does NOT depend on income; renters' does heavily
- Interpretation: Renters behave like rule-of-thumb consumers; homeowners closer to standard theory

### 6.8 Macro Expectations and Inflation Opinions (Table A.1)

**Full sample marginal effects on P(inf_lowbetter):**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| pi^e | +0.019 | (0.008) | ** |
| i^e_{savings} | -0.025 | (0.011) | ** |
| i^e_{mortgage} | +0.018 | (0.005) | *** |

Higher inflation expectations => more likely to view inflation as too high.
Higher mortgage rate expectations => more likely to prefer lower inflation.
In target zone [1.5, 2]: pi^e effect vanishes, but mortgage rate effect persists.

### 6.9 Macro Expectations and Interest Rate Opinions (Table A.2)

**Full sample marginal effects on P(int_highbetter):**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| pi^e | +0.001 | (0.004) | insig |
| i^e_{savings} | -0.033 | (0.007) | *** |
| i^e_{mortgage} | -0.007 | (0.003) | ** |

No inflation expectation effect on interest rate opinions; interest rate expectations dominate.
Higher savings/mortgage rate expectations => less likely to prefer higher rates.
Pattern is stable across target zone subsample.

---

## 7. CONSUMPTION GOODS AND HOUSING SPENDING (Tables A.3-A.4)

### 7.1 Consumption Goods (Table A.3)

**Full sample, OLS on log spending with interactions:**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| d_inf_highbetter | +0.377 | (0.133) | *** |
| d_int_lowbetter | +0.282 | (0.142) | ** |
| d_int_highbetter | +0.212 | (0.094) | ** |
| r^e_{sav} * d_inf_highbetter | +0.151 | (0.081) | * |
| r^e_{sav} * d_int_highbetter | +0.057 | (0.028) | ** |
| r^e_{sav} * d_int_lowbetter | +0.055 | (0.030) | * |

Opinions reduce the negative real-rate effect on consumption spending, rendering it insignificant.
Effects largely confined to full sample; vanish in target zone.

### 7.2 Housing Spending (Table A.4)

**Full sample OLS:**

| Variable | ME | SE | Sig |
|----------|-----|-----|-----|
| d_inf_lowbetter | +0.183 | (0.061) | *** |
| d_int_highbetter | -0.119 | (0.052) | ** |
| r^e_{savings} | +0.021 | (0.011) | ** |

Those preferring lower inflation spend MORE on housing; those preferring higher interest rates spend LESS.

---

## 8. MODEL FIT SUMMARY

### 8.1 Probit Models (Pseudo R-squared ranges)

| Model | Full Sample | Target Zone |
|-------|------------|-------------|
| Inflation opinions (demographics) | 0.047-0.055 | 0.076-0.089 |
| Interest rate opinions (demographics) | 0.019-0.087 | 0.031-0.165 |
| Inflation opinions (macro expectations) | 0.030-0.112 | 0.016-0.093 |
| Interest rate opinions (macro expectations) | 0.016-0.106 | 0.025-0.194 |

### 8.2 OLS Models (Adjusted R-squared ranges)

| Model | Full Sample | Target Zone |
|-------|------------|-------------|
| Durable spending | 0.015-0.041 | 0.076-0.125 |
| Savings | 0.145-0.163 | 0.171-0.236 |
| Consumption goods | 0.104-0.116 | 0.114-0.146 |
| Housing | 0.067-0.124 | 0.069-0.123 |

---

## 9. THEORETICAL CONNECTIONS

### 9.1 Intertemporal Substitution (Euler Equation)

Standard consumption Euler equation predicts:
- Higher real interest rate => postpone current consumption to future (b_2 < 0)
- Higher expected inflation => lower perceived real rate => increase current spending
- At zero lower bound (ZLB): inflation expectations become primary channel

Paper confirms this only for consumers with anchored expectations (target zone).

### 9.2 Loss Aversion / Asymmetric Transmission

The finding that consumers preferring lower interest rates have higher real-rate elasticity (-0.986 vs baseline) connects to:
- Yogo (2008): During contractions, elasticity of intertemporal substitution increases
- Rosenblatt-Wisch (2008): Loss aversion in aggregate macro time series
- Santoro et al. (2014): Asymmetric transmission of monetary policy

Interpretation: Consumers are more sensitive to real rate decreases than increases.

### 9.3 Taylor Rule Consistency

43% of sample simultaneously prefers lower inflation AND higher interest rates, consistent with a hawkish Taylor rule stance. Combined with ECB main refinancing rate at zero during survey period.

### 9.4 Attribution Theory (Behavioral Foundation)

Jones and Nisbett (1972): People view their own behavior as reflecting environmental demands.
Implication: Consumers who believe real rates SHOULD be lower act AS IF they ARE lower, increasing current spending. This provides the behavioral mechanism for the opinion channel.

---

## 10. METHODOLOGICAL NOTES

1. **Population weights** used in ALL estimations to correct for survey design
2. **Robust standard errors** throughout (heteroskedasticity-consistent)
3. **Outlier treatment**: Inflation expectations truncated [-5%, 25%]; interest rate expectations truncated [<=25%]; spending top 5% excluded
4. **Cross-wave identification**: One-month gap between waves assumed to not substantially change economic or personal conditions
5. **Non-linearity in probits**: Interaction effects in probit models require graphical marginal effect analysis (not direct coefficient interpretation)
6. **Reference categories**: "Appropriate" for opinions; full-time employment for work status; low education for education; low income for income
7. **Germany-specific context**: 53.6% renter share (OECD 2018); predominantly fixed-rate mortgages; ECB main refinancing rate = 0% during survey period; lingering East/West attitudinal differences from reunification
