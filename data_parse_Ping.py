import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ADC(Dataset):
    """A class that processes data of the type used in this paper.

    From the paper: "This study utilized clinical PK data from
    trastuzumab emtansine (T-DM1), a conjugated monoclonal antibody
    drug that has been approved for the treatment of patients
    with human epidermal growth factor receptor 2 (HER2)
    positive breast cancers (Boyraz et al., 2013).
    """

    def __init__(self, data, label_col, feature_cols):
        self.data = data

        self.label_col = label_col
        self.features = feature_cols
        # neural networks like inputs to be on similar scales
        # so normalizing into days here
        self.data.loc[:, "TIME"] = self.data["TIME"] / 24

    def __len__(self):
        return self.data.PTNM.unique().shape[0]

    def __getitem__(self, index):
        ptnm = self.data.PTNM.unique()[index]
        cur_data = self.data[self.data["PTNM"] == ptnm]
        times, features, labels, cmax_time = self.process(cur_data)
        return ptnm, times, features, labels, cmax_time

    def process(self, data):
        data = data.reset_index(drop=True)
        
        cmax_time = data.loc[data.TIME < 21, ["TIME", "PK_timeCourse"]].values.flatten()
        cmax = data.loc[data.TIME < 21, "PK_timeCourse"].max()


        # from the paper:
        # "We iterated TIME and PK values from the first time point
        # to the last time point during the first cycle, and padded
        # the vector to 20 elements long with zeros"
        cmax_time_full = np.zeros((20,))
        if len(cmax_time) <= 20:
            cmax_time_full[: len(cmax_time)] = cmax_time
        else:
            cmax_time_full[:] = cmax_time[:20]

        # normalizing PK_round1 to be on similar scales to other variables
        #data.loc[:, "PK_round1"] = data["PK_round1"] / cmax

        # note that PK_timeCourse (the label) is not normalized
        # but training may be easier if it was

        features = data[self.features].values
        labels = data[self.label_col].values
        times = data["TIME"].values

        times = torch.from_numpy(times)
        features = features.astype(np.float32)
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels).unsqueeze_(-1)
        cmax_time_full = torch.from_numpy(cmax_time_full)
        return times, features, labels, cmax_time_full


def adc_collate_fn(batch, device):
    """This function collates the elements of the batch
    into the format expected by the neural network
    """
    D = batch[0][2].shape[1]
    N = 1

    # from the paper: In the encoding step, a GRU layer with 128 hidden units scans through the whole time series
    # (5 channel input, TFDS, TIME, CYCL, AMT, and PK_cycle1, as in LSTM model) reversely (i.e., from the end to the start)
    combined_times, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_times = combined_times.to(device)

    combined_features = torch.zeros([len(batch), len(combined_times), D]).to(device)
    combined_label = torch.zeros([len(batch), len(combined_times), N]).to(device)
    combined_label[:] = np.nan
    combined_cmax_time = torch.zeros([len(batch), 20]).to(device)

    ptnms = []
    offset = 0
    for b, (ptnm, times, features, label, cmax_time) in enumerate(batch):
        ptnms.append(ptnm)

        times = times.to(device)
        features = features.to(device)
        label = label.to(device)
        cmax_time = cmax_time.to(device)

        indices = inverse_indices[offset : offset + len(times)]
        offset += len(times)

        combined_features[b, indices] = features.float()
        combined_label[b, indices] = label.float()
        combined_cmax_time[b, :] = cmax_time.float()
    combined_times = combined_times.float()

    return ptnms, combined_times, combined_features, combined_label, combined_cmax_time


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def parse_adc(device, train, validate, test, phase="train"):
    """This function constructs the dataset iterators that pytorch needs"""
    feature_cols = ["TFDS", "TIME", "AMT"]
    label_col = "PK_timeCourse"
    if phase == "train":
        train = ADC(train, label_col, feature_cols)
        validate = ADC(validate, label_col, feature_cols)
        ptnm, times, features, labels, cmax_time = train[0]

        train_dataloader = DataLoader(
            train, batch_size=1, shuffle=True, collate_fn=lambda batch: adc_collate_fn(batch, device)
        )
        val_dataloader = DataLoader(
            validate, batch_size=1, shuffle=False, collate_fn=lambda batch: adc_collate_fn(batch, device)
        )

        dataset_objs = {
            "train_dataloader": inf_generator(train_dataloader),
            "val_dataloader": inf_generator(val_dataloader),
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "input_dim": features.size(-1),
        }

    else:
        test = ADC(test, label_col, feature_cols)
        ptnm, times, features, labels, cmax_time = test[0]
        test_dataloader = DataLoader(
            test, batch_size=1, shuffle=False, collate_fn=lambda batch: adc_collate_fn(batch, device)
        )

        dataset_objs = {
            "test_dataloader": inf_generator(test_dataloader),
            "n_test_batches": len(test_dataloader),
            "input_dim": features.size(-1),
        }

    return dataset_objs
