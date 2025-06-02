import torch 

class DrugResponseModel_OneHot(torch.nn.Module):
    def __init__(self, input_size):
        super(DrugResponseModel_OneHot, self).__init__()

        self.full_cell_embed = torch.nn.Sequential(
                                torch.nn.Linear(input_size, 400),
                                torch.nn.ReLU(),
                                torch.nn.Linear(400, 300),
                                torch.nn.ReLU(),
                                torch.nn.Linear(300, 200),
                                torch.nn.ReLU(),
                                torch.nn.Linear(200, 100),
                                torch.nn.ReLU(),
                                torch.nn.Linear(100, 50),
                                torch.nn.ReLU(),
                                torch.nn.Linear(50, 25),
                                torch.nn.ReLU(),
                                )
        

        self.full_drug_embed = torch.nn.Sequential(
                                torch.nn.Linear(2048, 1000),
                                torch.nn.ReLU(),
                                torch.nn.Linear(1000, 500),
                                torch.nn.ReLU(),
                                torch.nn.Linear(500, 250),
                                torch.nn.ReLU(),
                                )
        
        

        self.calculate_response = torch.nn.Sequential(
                                    torch.nn.Linear(275,100),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(100,50),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(50,25),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(25,10),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(10,1)
                                    )
        
    def forward(self, x, d):

        x = self.full_cell_embed(x)
        d = self.full_drug_embed(d)
        combo = torch.cat((x,d), 1)
        pred_response = self.calculate_response(combo)

        return pred_response

class DrugResponseModel_Mini(torch.nn.Module):
    def __init__(self, input_size):
        super(DrugResponseModel_Mini, self).__init__()

        self.full_cell_embed = torch.nn.Sequential(
                                torch.nn.Linear(input_size, 1000),
                                torch.nn.ReLU(),
                                torch.nn.Linear(1000, 750),
                                torch.nn.ReLU(),
                                torch.nn.Linear(750, 500),
                                torch.nn.ReLU(),
                                torch.nn.Linear(500, 250),
                                torch.nn.ReLU(),
                                torch.nn.Linear(250, 100),
                                torch.nn.ReLU(),
                                torch.nn.Linear(100, 50),
                                torch.nn.ReLU(),
                                )
        

        self.full_drug_embed = torch.nn.Sequential(
                                torch.nn.Linear(2048, 100),
                                torch.nn.ReLU(),
                                torch.nn.Linear(100, 50),
                                torch.nn.ReLU(),
                                torch.nn.Linear(50, 25),
                                torch.nn.ReLU(),
                                )
        
        

        self.calculate_response = torch.nn.Sequential(
                                    torch.nn.Linear(75,50),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(50,25),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(25,1),
                                )

        
    
    def forward(self, x, d):

        x = self.full_cell_embed(x)
        d = self.full_drug_embed(d)
        combo = torch.cat((x,d), 1)
        pred_response = self.calculate_response(combo)
    
        return pred_response