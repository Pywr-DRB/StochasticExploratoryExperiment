# ROUGH TITLE: Exploratory Modeling to Better Understand How Changing Streamflow Seasonality May Shape Water Supply Vulnerabilities in the Delaware River Basin

# Questions

****
# Abstract
The Delaware River Basin (DRB) supplies water to over 14 million people, including New York City (NYC). The basin operates under a complex institutional framework designed to balance competing water use demands while maintaining United States Supreme Court-mandated streamflow targets designed to protect downstream users, water quality and ecosystems. During the 1960s drought of record, competing demands resulted in violations of both streamflow targets and diversion limits, creating significant tensions between the basin states. Since then, the DRB has not experienced comparable drought conditions, and mid-century climate projections indicate changes towards wetter on average annual streamflow conditions. However, the projected wetter annual trends do not account for the DRB’s internal variability or distinguish seasonal scale differences in how streamflows may change. These limitations raise two important questions: (1) what are the plausible future droughts that the DRB could confront? and (2) how vulnerable is the basin to those plausible future extremes? 
This study addresses these questions using a large ensemble-based stochastic exploratory analysis for the DRB. We develop two sets of streamflow scenarios, each composed of 1,000 70-year daily streamflow sequences. The first baseline scenario set captures the plausible drought extremes that emerge from the basin’s internal variability using a stationary synthetic ensemble based on historical flows. The second scenario set incorporates mid-century climate change projections in a climate-adjusted ensemble. The climate-adjusted ensemble accounts for projected shifts toward increased earlier peak runoff and reduced late-summer flows. We simulate performance of both ensembles using Pywr-DRB, a water resources systems model representing basin operations including NYC reservoirs, diversions, and downstream flow targets. Results reveal that the frequency of severe, 1960s-level drought events are twice as likely in the climate-adjusted ensemble relative to the stationary baseline. The climate-adjusted ensemble also results in a greater frequency of NYC reservoir depletion and flow target violation events. The findings demonstrate that changes to streamflow timing influence drought risk and system vulnerability in the DRB. 

## Key points

# 1.0 Introduction


# 2.0 Study Area: Delaware River Basin

# 3.0 Methodology


## 3.x Climate Change Data

- **CMIP data:** We draw from an ensemble of streamflow projections based on the latest Global Climate Model (GCM) projections from the Coupled Model Intercomparison Project phase 6 (CMIP6; Eyring et al., 2016). 
	- Shared Socioeconomic Pathways (SSPs) and Representative Concentration Pathways (RCPs).
	- These data have previously been used to evaluate the impacts of projected climate change on future U.S. hydropower generation (Kao et al., 2022). Summary of projected changes from Kao et al., 2022:
		- The emissions scenario has consistent influence on temperature but less clear impact on precipitation patterns across CONUS.  
		- Annual increase in precipitation is generally observed across the US, except during periods in summer and fall.
		- High runoff (95th percentile) is projected to increase in the majority of CONUS watersheds, however low runoff (5th percentile of 7 day average) is projected to decrease in many areas.
		- All the models considered in this study were statistically downscaled using the bias-corrected using the Double Bias Corrected Constructed Analogues approach (Werner and Cannon, 2016).
- **Subset of models used in this study with justification:** 

- **Monthly flow changes:** 


## 3.x Streamflow Generation


## 3.x Pywr-DRB


## 3.x Drought Metrics


For drought-event metrics $X_1, X_2$ \text{ on events }e\text{ with daily start/end timestamps, define the annual indicator}
$$
\mathbf{1}_{r,y}(x_1,x_2)=\mathbb{1}\{\exists e \text{ in realization } r \text{ that intersects calendar year } y \text{ with } X_1(e)\ge x_1,\ X_2(e)\ge x_2\}.
$$
With }R\text{ realizations and }Y=70\text{ years (same years for all realizations), the empirical frequency on grids }(x_1^{(i)},x_2^{(j)})\text{ is}
$$
\hat p_{ij}=\frac{1}{RY}\sum_{r=1}^R\sum_{y=1}^Y \mathbf{1}_{r,y}\!\big(x_1^{(i)},x_2^{(j)}\big),\qquad
\hat T_{ij}=\begin{cases}
1/\hat p_{ij}, & \hat p_{ij}>0\\
\infty, & \hat p_{ij}=0~.
\end{cases}
$$
We count an event for every calendar year it intersects. Multiple qualifying events in a year contribute at most 1 to the indicator (at-least-one rule).

### 2.x Operations Metrics


As used by Devineni and Ceylan (2014).

**Probability of reservoir refill every year on June 1st:**
$$P(S_t >= L2a) \text{ for } t = \text{June 1st}$$

**Probability of reservoir refill conditional on SSI drought**



Below are formal definitions matching the script’s estimator and counting rules.

$$
\textbf{Setup.} \quad
\mathcal{R}=\{1,\dots,R\}\ \text{(realizations)},\quad 
\mathcal{Y}=\{y_1,\dots,y_Y\}\ \text{(calendar years, common to all realizations)}.
$$

For realization $r\in\mathcal{R}$, let $\mathcal{E}_r$ be the set of drought events.
Each event $e\in\mathcal{E}*r$ has a start date $s_e$, end date $t_e$ (daily resolution), and metrics $X_1(e),X_2(e)\in\mathbb{R}*{\ge 0}$.
For year $y\in\mathcal{Y}$, let $\mathcal{I}_y$ denote the closed interval covering that calendar year.

$$
\textbf{Event–year intersection.} \quad 
e \pitchfork y \iff [s_e,t_e]\cap \mathcal{I}_y \neq \varnothing .
$$

$$
\textbf{Annual “at least one event” indicator at thresholds }(x_1,x_2). \quad
I_{r,y}(x_1,x_2)
:= \mathbf{1}\Big\{\exists\,e\in\mathcal{E}_r:\ e\pitchfork y,\ X_1(e)\ge x_1,\ X_2(e)\ge x_2\Big\}.
$$

Multiple qualifying events within the same $(r,y)$ contribute at most one.

$$
\textbf{Empirical joint exceedance frequency.} \quad
\hat p(x_1,x_2)
:= \frac{1}{RY}\sum_{r=1}^R\sum_{y\in\mathcal{Y}} I_{r,y}(x_1,x_2).
$$

This is the estimated probability that in a randomly selected realization–year pair, there exists at least one drought event intersecting that year with $X_1\ge x_1$ and $X_2\ge x_2$.

$$
\textbf{Return period (years).} \quad
\hat T(x_1,x_2)
:= \begin{cases}
\dfrac{1}{\hat p(x_1,x_2)}, & \hat p(x_1,x_2)>0,\\[8pt]
+\infty, & \hat p(x_1,x_2)=0.
\end{cases}
$$

In the script, a small regularization $\varepsilon:=1/(RY)$ may be applied via $\hat p_\varepsilon(x_1,x_2):=\max{\hat p(x_1,x_2),\varepsilon}$ before inversion to avoid infinite values when plotting.

$$
\textbf{Gridded representation.} \quad
\{x_{1,i}\}_{i=1}^{I},\ \{x_{2,j}\}_{j=1}^{J}\ \text{(linear grids)},\quad
\hat p_{ij}:=\hat p(x_{1,i},x_{2,j}),\ \hat T_{ij}:=\hat T(x_{1,i},x_{2,j}).
$$

By construction, $\hat p_{ij}$ is nonincreasing in each threshold: if $i'\<i$ or $j'\<j$, then $\hat p_{i'j}\ge \hat p_{ij}$ and $\hat p_{ij'}\ge \hat p_{ij}$.

$$
\textbf{Scenario comparison (percent change in return period).} \quad
\text{For two scenarios } s,c \text{ with frequencies } \hat p_s,\hat p_c,
\text{ the script computes}
$$

$$
\Delta_T(x_1,x_2)
:= 100\left(\frac{\hat p_s(x_1,x_2)}{\hat p_c(x_1,x_2)} - 1\right),
$$

which equals $100\big(\hat T_c/\hat T_s - 1\big)$ whenever both return periods are finite. Positive $\Delta_T$ indicates a longer (rarer) return period under scenario $c$; negative $\Delta_T$ indicates a shorter (more frequent) return period. For numerical robustness in zero–probability cells, $\hat p$ may be $\varepsilon$–regularized before forming the ratio.


# 4.0 Results and Discussion




# 5.0 Conclusions

- Assumptions:
	- Applied the same percentage monthly streamflow across all model nodes. Whereas the changes in streamflow patterns would likely differ across the basin.
- Limitations
	- Changes in reservoir evaporative losses were not considered in this study.

