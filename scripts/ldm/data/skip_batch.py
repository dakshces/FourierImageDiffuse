from torch.utils.data import Dataset

class SkippingDataset(Dataset):
    def __init__(self, dataset, skip_probability=0.1):
        """
        Wraps a dataset, probabilistically skipping samples.
        :param dataset: Dataset instance (wrapped)
        :param skip_probability: Proportion of samples to skip
        """
        self.dataset = dataset
        self.skip_probability = skip_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import random
        # Skip batch logic
        while random.random() < self.skip_probability:
            index = (index + 1) % len(self.dataset)  # Avoid running out of index
        return self.dataset[index]
