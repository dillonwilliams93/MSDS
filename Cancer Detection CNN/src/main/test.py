# run the saved model on the test data
# local functions
from classes import cnn, img_dataset
from functions import transform
# external functions
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Initiate model
model = cnn.CNN()

# Load saved model
model.load_state_dict(torch.load('/Users/dillonwilliams/pycharmprojects/Cancer Detection/results/models/model-4.pth'))
model.eval()

# load transformer
transformer = transform.transformer()

# load test data
test_data = img_dataset.CustomImageDataset('/Users/dillonwilliams/pycharmprojects/Cancer Detection/data/test',
                                           transform=transformer)

# create the dataloader
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# initiate prediction and label lists
predictions, ids = [], []

# run the model on the test data
with torch.no_grad():
    for inputs, _, img_id in test_loader:
        # get outputs
        outputs = model(inputs)

        # sigmoid outputs
        outputs = torch.sigmoid(outputs)

        # apply threshold
        prediction = (outputs > 0.5).cpu().numpy()

        # append to lists
        predictions.append(1 if prediction else 0)
        ids.append(img_id[0].split('.')[0])

# Convert results to a pandas DataFrame
results_df = pd.DataFrame({
    'id': ids,
    'label': predictions
})

# Save the results
results_df.to_csv('/Users/dillonwilliams/pycharmprojects/Cancer Detection/results/predictions/model-4.csv',
                  index=False)