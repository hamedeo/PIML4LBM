import numpy as np

# Load the .npz file containing your training data
data = np.load('saved_data/my_bgk_data.npz')
f_pre = data['f_pre']
f_post = data['f_post']

# Save the arrays to CSV files
np.savetxt('f_pre.csv', f_pre, delimiter=',', header="f_pre", comments='')
np.savetxt('f_post.csv', f_post, delimiter=',', header="f_post", comments='')

print("Data has been converted to CSV files: f_pre.csv and f_post.csv")
