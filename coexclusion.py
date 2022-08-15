import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# Presence/Absence determined with fixed threshold
def counts_to_presence_absence(df, abundance_thresh):
    return df > abundance_thresh

# Presence/Absence determined with dynamic threshold per reference defined as xth percentile
# among samples that pass the min_abundance_threshold
def counts_to_dynamic_presence_absence(df, min_abundance_thresh):
    new_cols = []
    for c in df.columns:
        s = df[c]
        s = s[s > min_abundance_thresh]
        dyn_thresh = s.quantile(.1)
        print(c,dyn_thresh)
        
        new_cols.append(df.loc[:, c] > dyn_thresh)
    data = pd.DataFrame(new_cols)
    return data.T
        

def presence_absence_to_contingency(pa):
    full_table = []
    for ci in pa.columns:
        active_row = []
        for cj in pa.columns:            
            crosstab = pd.crosstab(pa[ci], pa[cj])
            active_row.append(crosstab)
        full_table.append(pd.concat(active_row, axis=1))
    
    final = pd.concat(full_table, axis=0)
    names = []
    for c in pa.columns:
        names.append(c + "_Absent")
        names.append(c + "_Present")
    final.columns = names
    final.index = names
    return final

def pairwise_eval(pa):
    for ci in pa.columns:
        for cj in pa.columns:
            crosstab = pd.crosstab(pa[ci], pa[cj])
            c, p, dof, expected = chi2_contingency(crosstab)
            print(ci, cj, "chisq=" + str(c),"p="+str(p))                
            if p > 0.001 and p < 0.05:
                print("Observed")
                print(crosstab)
                print("Expected")
                print(np.round_(expected))

    

if __name__=="__main__":
    counts_df = pd.DataFrame([[250, 500, 750], [125, 250, 375], [10000, 20000, 30000]], columns=["G_A", "G_B", "G_C"])
    print(counts_df)

    pa = counts_to_presence_absence(counts_df, 500)
    print(pa)
    dpa = counts_to_dynamic_presence_absence(counts_df, 10)
    print(dpa)
    
    contingency = presence_absence_to_contingency(pa)
    print(contingency)
    
    pairwise_eval(pa)