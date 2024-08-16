import streamlit as st
import pandas as pd
import numpy as np
import random

# """
# AI 
# """

# class for evaluate
class distance_on_performance:
    def __init__(self, evaluation_matrix, weight_matrix, criteria):
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")
        self.row_size = len(self.evaluation_matrix)
        self.column_size = len(self.evaluation_matrix[0])
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.criteria = np.array(criteria, dtype="float")

    def step_2(self):
        sqrd_sum = np.sqrt(np.sum(self.evaluation_matrix**2, axis=0))
        self.normalized_decision = self.evaluation_matrix / sqrd_sum[np.newaxis, :]

    def step_3(self):
        self.weighted_normalized = self.normalized_decision * self.weight_matrix

    def step_4(self):
        self.worst_alternatives = np.min(self.weighted_normalized, axis=0)
        self.best_alternatives = np.max(self.weighted_normalized, axis=0)
        
        # Adjust based on criteria (if False, we want the minimum)
        self.worst_alternatives = np.where(self.criteria, self.worst_alternatives, self.best_alternatives)
        self.best_alternatives = np.where(self.criteria, self.best_alternatives, self.worst_alternatives)

    def step_5(self):
        self.worst_distance = np.sqrt(np.sum((self.weighted_normalized - self.worst_alternatives)**2, axis=1))
        self.best_distance = np.sqrt(np.sum((self.weighted_normalized - self.best_alternatives)**2, axis=1))

    def step_6(self):
        self.worst_similarity = self.worst_distance / (self.worst_distance + self.best_distance)
        self.best_similarity = self.best_distance / (self.worst_distance + self.best_distance)

    def ranking(self, data):
        return np.argsort(np.argsort(data)) + 1

    def rank_to_worst_similarity(self):
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        return self.ranking(self.best_similarity)

    def calc(self):
        st.write("Step 1: Raw data")
        st.write(self.evaluation_matrix)
        self.step_2()
        st.write("Step 2: Normalized Matrix")
        st.write(self.normalized_decision)
        self.step_3()
        st.write("Step 3: Calculate the weighted normalized decision matrix")
        st.write(self.weighted_normalized)
        self.step_4()
        st.write("Step 4: Determine the worst alternative and the best alternative")
        st.write(f"Worst: {self.worst_alternatives}")
        st.write(f"Best: {self.best_alternatives}")
        self.step_5()
        st.write("Step 5: Distance from Best to Worst")
        st.write(f"Worst distance: {self.worst_distance}")
        st.write(f"Best distance: {self.best_distance}")
        self.step_6()
        st.write("Step 6: Similarity")
        st.write(f"Worst similarity: {self.worst_similarity}")
        st.write(f"Best similarity: {self.best_similarity}")

def assign_weights(df, source_columns, base_weights):
    
    def calculate_individual_weights(row):
        individual_weights = []
        for source in source_columns:
            if pd.notna(row[source]):
                individual_weights.append(base_weights.get(row[source], 0.1))
            else:
                individual_weights.append(0)
    
        # Normalize weights to sum to 1
        total_weight = sum(individual_weights)
        st.write(f"1. Individual weight: {individual_weights}")
        st.write(f"2. Total weight: {round(total_weight, 4)}")
        st.write(f"3. Normalization weight: {[w / total_weight for w in individual_weights]}")
        st.write(f"4. Total Normalization weight: {round(sum([w / total_weight for w in individual_weights]), 4)}")
        st.markdown("***")

        if total_weight > 0:
            return [w / total_weight for w in individual_weights]
        else:
            return [1/len(source_columns)] * len(source_columns)  # Equal weights if no valid sources

    weights = df.apply(calculate_individual_weights, axis=1)
    return np.array(weights.tolist())

def prepare_data_for_topsis(df, source_columns, total_columns, base_weights):    
    scores = df[total_columns].values
    weights = assign_weights(df, source_columns, base_weights)
    
    return scores, weights

def tier_rank(tier_label, quantile, target_columns):
    results, bin_edges = pd.qcut(target_columns,
                            q=quantile,
                            labels=tier_label,
                            retbins=True)

    tier_table = pd.DataFrame(zip(bin_edges, tier_label),
                                columns=['Threshold', 'Tier'])
    
    return tier_table

def normalization_pms_toptalent(df, criteria, source_columns, total_columns, base_weights):
    st.write("Step 0: assign weight")
    scores, weights = prepare_data_for_topsis(df, source_columns, total_columns, base_weights)
    
    top = distance_on_performance(scores, weights, criteria)
    top.calc()
    
    members = df['PERSON_ID'].tolist()
    result = 1 - top.best_similarity

    arr = sorted(zip(members, result), key=lambda x: x[1], reverse=True)
    reg_df = pd.DataFrame(arr, columns=['PERSON_ID', 'TALENT_SCORE'])

    return reg_df

def toptalent_dynamic_weight(df, criteria, scores_columns, total_columns, base_weights):
    reg_df = normalization_pms_toptalent(df, criteria, scores_columns, total_columns, base_weights)

    bin_labels = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    q = [0, .4, .6, .8, .9, 1]

    tier_table = tier_rank(bin_labels, q, reg_df['TALENT_SCORE'])
    tier_table['Percentile'] = q[:-1]

    bins = list(tier_table['Threshold']) + [float('inf')]
    labels = tier_table['Tier']

    reg_df['TIER'] = pd.cut(reg_df['TALENT_SCORE'], bins=bins, labels=labels, right=False)

    toptalent = reg_df.copy(deep=True)
    tier_order = ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze']
    toptalent['TIER'] = pd.Categorical(toptalent['TIER'], categories=tier_order, ordered=True)
    toptalent = toptalent.sort_values('TIER')

    return toptalent, tier_table


# """
# APPICATION
# """

# Define job families and sources
job_families = ["Energy Technology - Sales", "Human Resources Management", "Strategic Planning - Finance strategy planning", "Digital & Innovation", "Commercial - Business Development"]
sources = ["Direct Manager", "Other Manager", "Peer within Function", "Self", "External", "Peer across Department", "Subordinate"]

# Initialize the DataFrame structure
columns = [
    "PERSON_ID", "JOB_FAMILY", 
    "SOURCE_SOURCE1", "SOURCE_SOURCE2", "SOURCE_SOURCE3", 
    "SOURCE_SOURCE4", "SOURCE_SOURCE5", "SOURCE_SOURCE6", "SOURCE_SOURCE7", 
    "TOTAL_SOURCE1", "TOTAL_SOURCE2", "TOTAL_SOURCE3", 
    "TOTAL_SOURCE4", "TOTAL_SOURCE5", "TOTAL_SOURCE6", "TOTAL_SOURCE7"
]
data = pd.DataFrame(columns=columns)

# Streamlit app title
st.title("Employee Score Input Simulation")
st.subheader(f"How many employees do you want to compare?")
num_emp = st.number_input(f"Number of employee", min_value=0, max_value=100, value = 3)

# Input form for 5 individuals
st.header(f"Step1: Input Example employees")
for i in range(num_emp):
    st.subheader(f"Employee {i+1}")
    
    person_id = st.text_input(f"Person ID {i+1}", f"ID{i+1}")
    job_family = st.selectbox(f"Job Family {i+1}", job_families)
    
    st.subheader("Score Sources")
    selected_sources = [st.selectbox(f"Source {j+1} for Employee {i+1}", sources) for j in range(7)]
    
    st.subheader("Total Scores")
    # Generate random scores for each source
    score_inputs = random.uniform(0.0, 60.0)

    total_scores = [st.number_input(f"Total Score for Source {j+1} (Employee {i+1})", min_value=0.0, max_value=60.0, value = score_inputs) for j in range(7)]
    
    # Create a new DataFrame for the current employee
    new_row = pd.DataFrame([[person_id, job_family] + selected_sources + total_scores], columns=columns)
    
    # Concatenate the new row to the existing DataFrame
    data = pd.concat([data, new_row], ignore_index=True)

# Display the resulting DataFrame
st.write("Resulting DataFrame:")
st.dataframe(data)

# Line break
st.markdown("***")

st.header(f"Step2: Input weight")

# Initialize dictionary for weights
base_weights = {}

# Input fields for each source
for source in sources:
    weight = st.number_input(f"Weight for {source}", min_value=0.0, max_value=1.0, value=0.1)
    base_weights[source] = weight

# Display the weights
st.write("Base Weights Dictionary:")
st.json(base_weights)

# Line break
st.markdown("***")

st.header(f"Step3: Calm down and see the result🥹")
criteria = [True] * 7  # 7 sources
source_columns = ['SOURCE_SOURCE1', 'SOURCE_SOURCE2', 'SOURCE_SOURCE3', 'SOURCE_SOURCE4', 'SOURCE_SOURCE5', 'SOURCE_SOURCE6', 'SOURCE_SOURCE7']
total_columns = ['TOTAL_SOURCE1', 'TOTAL_SOURCE2', 'TOTAL_SOURCE3', 'TOTAL_SOURCE4', 'TOTAL_SOURCE5', 'TOTAL_SOURCE6', 'TOTAL_SOURCE7']

if st.button("Calculate Top Talent"):
    with st.spinner("Calculating..."):
        st_output = st.empty()  # Placeholder to capture print outputs
        result_df, tier_table = toptalent_dynamic_weight(data, criteria, source_columns, total_columns, base_weights)
    
    st.success("Calculation Complete!")
    st.subheader("Top Talent Result")
    st.dataframe(result_df)
    st.subheader("Tier Table")
    st.dataframe(tier_table)