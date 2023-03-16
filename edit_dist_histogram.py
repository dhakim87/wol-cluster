from matplotlib import pyplot as plt

data = {'G000183345_NM:i:1': 1731129, 'G000012005_NM:i:0': 589828, 'G000183345_NM:i:0': 3831054, 'G000012005_NM:i:1': 873560, 'G000183345_NM:i:3': 486646, 'G000012005_NM:i:2': 909897, 'G000183345_NM:i:5': 232497, 'G000012005_NM:i:6': 362087, 'G000012005_NM:i:5': 561801, 'G000012005_NM:i:3': 861187, 'G000183345_NM:i:4': 324064, 'G000183345_NM:i:2': 836798, 'G000012005_NM:i:4': 719128, 'G000012005_NM:i:7': 224320, 'G000183345_NM:i:7': 85461, 'G001941055_NM:i:1': 36505, 'G000183345_NM:i:6': 142358, 'G001941055_NM:i:0': 110570, 'G001941055_NM:i:7': 2040, 'G001941055_NM:i:4': 6036, 'G001941055_NM:i:5': 4950, 'G001941055_NM:i:3': 9336, 'G001941055_NM:i:2': 15457, 'G001941055_NM:i:6': 3267}

parsed_data = {}
ecoli = "G000183345"
shigella = "G000012005"
ruminococcus = "G001941055"

data_ecoli = {}
data_shigella = {}
data_ruminococcus = {}
for key in data:
    count = int(key[len("G000183345_NM:i:")])

    if key.startswith(ecoli):
        data_ecoli[count] = data[key]
    elif key.startswith(shigella):
        data_shigella[count] = data[key]
    elif key.startswith(ruminococcus):
        data_ruminococcus[count] = data[key]

print(data_ecoli)
print(data_shigella)
print(data_ruminococcus)

total_ecoli = 0
total_shigella = 0
total_ruminococcus = 0
for edit_dist in range(8):
    total_ecoli += data_ecoli[edit_dist]
    total_shigella += data_shigella[edit_dist]
    total_ruminococcus += data_ruminococcus[edit_dist]

print("Total Shigella")
print(total_shigella)
print("Total Ecoli")
print(total_ecoli)
print("Total Ruminococcus")
print(total_ruminococcus)


fig, axes = plt.subplots(2, 2, sharey=True)

def plot_cumulative(ax, title, data_src, total_src):
    print(title)
    xs = []
    heights = []
    cumulative = 0
    for edit_dist in range(8):
        cumulative += data_src[edit_dist]
        print(edit_dist, "{:.3f}".format(cumulative / total_src))
        xs.append(edit_dist)
        heights.append(cumulative/total_src * 100)
    ax.bar(xs, heights)
    ax.set_title(title)
    ax.set_xlabel("Edit Distance (ED)")
    ax.set_ylabel("Assigned Reads < ED (%)")

plot_cumulative(axes[0][0], "Escherichia coli\n(G000183345)", data_ecoli, total_ecoli)
plot_cumulative(axes[1][0], "Shigella Dysenteriae\n(G000012005)", data_shigella, total_shigella)
plot_cumulative(axes[0][1], "Ruminococcus Zagget7\n(G001941055)", data_ruminococcus, total_ruminococcus)

plt.tight_layout()
plt.show()