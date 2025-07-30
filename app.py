import os
import json
import pandas as pd
import re
import csv
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re, json
from collections import defaultdict

@st.cache_data(show_spinner=True)
def parse_criteria_data(data):
    pattern = r'\{"criterion \d+":'
    max_criteria = 100
    all_criterion = pd.DataFrame()

    for idx, row in data.iterrows():
        rubric = row.get('Rubric Only')
        try:
            rubric_json = json.loads(rubric)
            flattened = []
            for item in rubric_json:
                for criterion_id, details in item.items():
                    entry = {'criterion_id': criterion_id}
                    entry.update(details)
                    flattened.append(entry)
            rubric_df = pd.DataFrame(flattened)
            criterion_ids = rubric_df['criterion_id'].tolist()
            id_numbers = [cid.split()[-1] for cid in criterion_ids]
            id_to_index = {id_num: i for i, id_num in enumerate(id_numbers)}
            n = len(id_numbers)
            grid = [['0'] * n for _ in range(n)]
            for i, deps in enumerate(rubric_df['dependent_criteria']):
                for dep in deps:
                    dep_str = str(dep)
                    if dep_str in id_to_index:
                        j = id_to_index[dep_str]
                        grid[i][j] = '1'
            dependency_grid = pd.DataFrame(grid, columns=id_numbers, index=id_numbers)
            dependency_grid_clean = dependency_grid.astype(int)
            dependency_grid_clean.index = dependency_grid_clean.index.astype(str)
            dependency_grid_clean.columns = dependency_grid_clean.columns.astype(str)

            def get_depth(criterion, visited=None):
                if visited is None:
                    visited = set()
                if criterion in visited:
                    return 0
                visited.add(criterion)
                deps = dependency_grid_clean.loc[criterion]
                direct = [col for col in dependency_grid_clean.columns if deps[col] == 1]
                return 0 if not direct else 1 + max([get_depth(d, visited.copy()) for d in direct])

            depths = {crit: get_depth(crit) for crit in dependency_grid_clean.index}

            current_cols = dependency_grid.shape[1]
            if current_cols < max_criteria:
                existing_cols = list(dependency_grid.columns)
                all_cols = existing_cols + [str(i) for i in range(current_cols + 1, max_criteria + 1)]
                for col in all_cols:
                    if col not in dependency_grid.columns:
                        dependency_grid[col] = pd.NA

            dependency_grid_c = dependency_grid.apply(pd.to_numeric, errors='coerce')
            dependency_grid['criterion_sum'] = dependency_grid_c.sum(axis=1, skipna=True, min_count=1)
            renamed_columns = {
                col: f"criterion_{col}" for col in dependency_grid.columns if col != 'criterion_sum'
            }
            dependency_grid = dependency_grid.rename(columns=renamed_columns)
            cols = ['criterion_sum'] + [col for col in dependency_grid.columns if col != 'criterion_sum']
            dependency_grid = dependency_grid[cols]

            rubric_df = rubric_df.reset_index(drop=True)
            dependency_grid = dependency_grid.reset_index(drop=True)
            dependency_grid.insert(1, 'criterion_dependency_depth', list(depths.values()))
            rubric_df = pd.concat([rubric_df, dependency_grid], axis=1)
            rubric_df.insert(0, 'original_rubric', row['Rubric Only'])
            rubric_df.insert(0, 'ID', row['ID'])

            all_criterion = pd.concat([all_criterion, rubric_df], axis=0)

        except Exception:
            continue

    return all_criterion



st.set_page_config(layout="wide")
st.title("🧠 DAG Viewer from 'Rubric Only' JSON")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if not uploaded_file:
    st.stop()

data = pd.read_csv(uploaded_file)
data = data.dropna(subset=['ID'])
data = data[data['ID'] != 'NaN']  
all_criterion = parse_criteria_data(data)

# ------ Read in file ----------- #

# What is the maximum number of criteria? We need this for padding out the values
pattern = r'\{"criterion \d+":'
max_criteria = max(data['Rubric Only'].fillna('').apply(lambda x: len(re.findall(pattern, x))))
max_criteria = 100 # make it arbitrarily large
print("the number of criteria to pad for: " + str(max_criteria) + '\n')

all_criterion = pd.DataFrame()


# Iterate + get the rubric-specific values
for idx, row in data.iterrows():
    #print(idx)

    rubric = row['Rubric Only']
    
    try:
        rubric_json = json.loads(rubric)
        #print(rubric_json)

        # Flatten JSON into long-format
        flattened = []
        for item in rubric_json:
            for criterion_id, details in item.items():
                entry = {'criterion_id': criterion_id}
                entry.update(details)
                flattened.append(entry)
        
        # Convert to DataFrame
        rubric_df = pd.DataFrame(flattened)

        # Turn into a dependency graph
        criterion_ids = rubric_df['criterion_id'].tolist()
        id_numbers = [cid.split()[-1] for cid in criterion_ids]  # ['1', '2', '3', '4', '5']
        
        # Step 2: Map from ID to row index
        id_to_index = {id_num: i for i, id_num in enumerate(id_numbers)}
        
        # Step 3: Create empty 0-grid
        n = len(id_numbers)
        grid = [['0'] * n for _ in range(n)]
        
        # Step 4: Fill in the grid based on dependent_criteria
        for i, deps in enumerate(rubric_df['dependent_criteria']):
            for dep in deps:
                dep_str = str(dep)
                if dep_str in id_to_index:
                    j = id_to_index[dep_str]
                    grid[i][j] = '1'
        
        dependency_grid = pd.DataFrame(grid, columns=id_numbers, index=id_numbers)
        #print(dependency_grid)



        ##### Calculate depth of dependencies (i.e., dependencies of dependencies)
        dependency_grid_clean = dependency_grid.astype(int)

        # Convert index and columns to strings for consistent lookup
        dependency_grid_clean.index = dependency_grid_clean.index.astype(str)
        dependency_grid_clean.columns = dependency_grid_clean.columns.astype(str)
        
        # Recursive function to calculate depth
        def get_depth(criterion, visited=None):
            if visited is None:
                visited = set()
            if criterion in visited:
                return 0  # prevent cycles
            visited.add(criterion)
        
            deps = dependency_grid_clean.loc[criterion]
            direct = [col for col in dependency_grid_clean.columns if deps[col] == 1]
            
            if not direct:
                return 0
            else:
                return 1 + max([get_depth(d, visited.copy()) for d in direct])
        
        # Compute depth for each criterion
        depths = {
            crit: get_depth(crit) for crit in dependency_grid_clean.index
        } # We'll add this to dependency_grid later 


        
        
        #### Pad out rows (optional)
        current_cols = dependency_grid.shape[1]
        target_cols = max_criteria

        if current_cols < target_cols:
            existing_cols = list(dependency_grid.columns)
            all_cols = existing_cols + [str(i) for i in range(current_cols + 1, target_cols + 1)]
            for col in all_cols:
                if col not in dependency_grid.columns:
                    dependency_grid[col] = pd.NA



        
        #### Get row totals
        dependency_grid_c = dependency_grid.apply(pd.to_numeric, errors='coerce')
        dependency_grid['criterion_sum'] = dependency_grid_c.sum(axis=1, skipna=True, min_count=1)
        
        ### Rename and reorder to be more logical + readable
        renamed_columns = {
            col: f"criterion_{col}" for col in dependency_grid.columns if col != 'criterion_sum'
        }
        dependency_grid = dependency_grid.rename(columns=renamed_columns)
        
        cols = ['criterion_sum'] + [col for col in dependency_grid.columns if col != 'criterion_sum']
        dependency_grid = dependency_grid[cols]


        
        ### Combine dfs to get one output
        rubric_df = rubric_df.reset_index(drop=True)
        dependency_grid = dependency_grid.reset_index(drop=True)
        dependency_grid.insert(1, 'criterion_dependency_depth', list(depths.values()))
        
        rubric_df = pd.concat([rubric_df, dependency_grid], axis=1)
        
        #### Tidyup
        rubric_df.insert(0, 'original_rubric', row['Rubric Only']) # massively increases file size
        #rubric_df.insert(0, 'Domain', row['Domain'])
        rubric_df.insert(0, 'ID', row['ID'])

        all_criterion = pd.concat([all_criterion,
                                  rubric_df], axis=0)
        
        del dependency_grid_c
        
    
    except:
        continue
        #print(str(idx) + ': woops')


st.success("✅ Dependencies parsed!")









# --- DAG Plotting ---
st.header("📈 DAG Plots by Task ID")


# Ensure Graphviz binary path is available
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# --- Load and filter ---
task_ids = sorted(all_criterion["ID"].unique())
selected_id = st.selectbox("Select Task ID", task_ids)

df_task = all_criterion[all_criterion["ID"] == selected_id].copy()


# --- Clean criterion_id: remove prefix and colons ---
def clean_node_id(val):
    return str(val).replace("criterion ", "").replace(":", "").strip()

df_task["node_id"] = df_task["criterion_id"].apply(clean_node_id)

# --- Clean dependent_criteria column ---
df_task["dependent_criteria"] = df_task["dependent_criteria"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
)

# --- Build graph ---
G = nx.DiGraph()
valid_nodes = set(df_task["node_id"])

for _, row in df_task.iterrows():
    node_id = row["node_id"]
    G.add_node(node_id)

for _, row in df_task.iterrows():
    current = row["node_id"]
    deps = row.get("dependent_criteria", [])
    if isinstance(deps, list):
        for dep in deps:
            dep_id = clean_node_id(dep)
            if dep_id in valid_nodes:
                G.add_edge(dep_id, current)

print(f"✅ Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
if not nx.is_directed_acyclic_graph(G):
    print("⚠️ Not a DAG! Cycles found:")
    for cycle in nx.simple_cycles(G):
        print("  ⮕", " → ".join(cycle))
else:
    print("✅ DAG is valid")

# --- Compute depth (longest path to node) ---
depths = {}
for node in nx.topological_sort(G):
    preds = list(G.predecessors(node))
    if not preds:
        depths[node] = 0
    else:
        depths[node] = max([depths[pred] for pred in preds]) + 1

# --- Group nodes by depth ---
layers = defaultdict(list)
for node, depth in depths.items():
    layers[depth].append(node)

# --- Build manual layout: x = depth, y = spacing ---
# --- Improved layout: x = depth, y = centered and spaced ---
pos = {}
layer_y_spacing = 1.5
for depth_level, nodes in layers.items():
    n_nodes = len(nodes)
    total_height = (n_nodes - 1) * layer_y_spacing
    for i, node in enumerate(sorted(nodes, key=lambda x: int(x))):
        x = depth_level * 0.02
        y = total_height / 2 - i * layer_y_spacing
        pos[node] = (x, y)

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 7))

# --- Color nodes: standalone = light pink, others = dodgerblue ---
standalone_nodes = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0]
connected_nodes = [n for n in G.nodes if n not in standalone_nodes]

# Draw standalone nodes (pink)
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    nodelist=standalone_nodes,
    node_size=1500,
    node_color="lightpink"
)

# Draw connected nodes (blue)
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    nodelist=connected_nodes,
    node_size=1500,
    node_color="dodgerblue"
)
nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=50, edge_color="gray", connectionstyle='arc3,rad=0.05')
nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold", font_color="white")

plt.title(f"DAG for Task ID {selected_id}", fontsize=16)
plt.axis("off")
plt.tight_layout()

st.pyplot(fig)
