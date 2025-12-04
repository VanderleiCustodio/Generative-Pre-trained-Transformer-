import requests

response = requests.get(url="https://lolbas-project.github.io/api/lolbas.json", verify=False)
ScratchData = ''
for value in response.json():    
    for item in value['Commands']:
        Name = value['Name']
        Command = item['Command']
        Description = item['Description']
        Usecase = item['Usecase']
        
        ScratchData += ''.join(f"""
Name : {Name}
Command : {Command}
Description : {Description}
Usecase : {Usecase}
                               """)
        
            
with open('lolbas.txt','w') as f:
    f.write(ScratchData)