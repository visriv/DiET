class Dataset(torch.utils.data.Dataset):



    def __init__(self, data, labels):

        self.data = data

        self.labels = labels



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        return idx, self.data[idx], self.labels[idx]


class DatasetfromDisk(torch.utils.data.Dataset):



    def __init__(self, data, labels):

        self.data = data

        self.labels = labels

        target_resolution = (224, 224)

        self.transform = transforms.Compose([

                    transforms.Resize(target_resolution),

                    transforms.ToTensor(),

            ])



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        image = self.transform(Image.open(self.data[idx]).convert('RGB'))

        

        return idx, image, self.labels[idx]
 

class DatasetwPreds(torch.utils.data.Dataset):



    def __init__(self, data, labels, model_original_preds, load_upfront=False):



        self.data = data

        self.labels = labels

        self.load_upfront = load_upfront



        target_resolution = (224, 224)

        self.transform = transforms.Compose([

                    transforms.Resize(target_resolution),

                    transforms.ToTensor(),

                ])

        self.model_original_preds = model_original_preds



        if load_upfront == True:

            self.data = torch.zeros((len(data), 3, 224, 224))

            for i, path in enumerate(data):

                image = Image.open(path).convert('RGB')

                image = self.transform(image)

                self.data[i] = image





    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        image = self.data[idx]

        

        if self.load_upfront == False:

            image = Image.open(image).convert('RGB')

            image = self.transform(image)



        return idx, image, self.labels[idx], self.model_original_preds[idx]

