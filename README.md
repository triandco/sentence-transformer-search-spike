# Introduction
This project attempts to test [sentence-transformers](https://www.sbert.net) as a means to generate search indexes for paragraphs inside documents as well as large documents.


# Requirements
* Python 3.7.9


# Run the project
This project use [virtualenv](https://docs.python.org/3/library/venv.html)

1. Start virtual environment
```powershell
python -m venv env
env/Script/active
```

2. Install [Pytorch for your platform](https://pytorch.org/). 
If you are on Windows and have an NVIDIA graphic card:
```powershell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

3. Install sentence-transformer
```powershell
pip install sentence-transformer
```

# Outcome

## Paragraph search
### Search for "3 principles" returns somewhat close results
Found 5
```
And Will:       
 (Score: 73.0641)
```
```
3. They timestamp their vision, whether in public announcements or internal documents.
 (Score: 65.0560)
```
```
So the third leg of the strategy is to apply Capital’s taste to the products it builds.
 (Score: 55.3959)
```
```
3. Jordi laid out the plan that they’ve been executing 
against in the first conversation we had. While some minor details have changed, the vision and the broad strokes of the strategy have remained consistent.
 (Score: 54.1502)
```
```
And Devin:
 (Score: 50.6368)
```
### Search for "3 approaches" returns a satisfactory result
Found 5
```
Capital’s answer centers on three interlinked strategies:   
 (Score: 69.3091)
```
```
Co-founders Jordi Hays and Sarah Chase are exceptional and complement each other well. 
 (Score: 60.2648)
```
```
So the third leg of the strategy is to apply Capital’s taste to the products it builds.
 (Score: 58.5801)
```
A more realistic upside comp set are the banks and fintechs 
focused on startups, including Silicon Valley Bank, Brex, Ramp, AngelList, and Mercury. While these are best-in-class companies (and private valuations were all set before the downturn), and there’s nothing close to a guarantee that Capital will reach their heights, it paints the size of the upside 
if Capital pulls off its ambitious plan. It also illustrates that while it will face competition, this space is large enough that there have, can be, and will continue to be many huge winners.
 (Score: 52.7117)

That’s the bet that husband and wife co-founders Jordi Hays 
and Sarah Chase have been making from day one, when it seemed crazy. But the insight is sneaky brilliant, grounded in the reality of the competitive landscape.
 (Score: 50.2830)



### Search for 'younger generation' returns satisfactory result
Found 5
```
The idea is simple but powerful: if you can acquire the youngest users, retain them as they grow up, and continue to attract the new cohorts of young users, you will win over time.
 (Score: 66.8162)
```
```
Leverage the Compounding Power of Young Users
 (Score: 59.1707)
```
```
•       Leverage the Compounding Power of Young Users       
 (Score: 56.7534)
```
```
•       Leverage the Compounding Power of Young Users       
 (Score: 56.7534)
```
Jordi is the kind of founder I call a Worldbuilder. Worldbuilders all have a few things in common:
 (Score: 42.6172)

### Search for 'extending its taste' return satisfactory result:
Found 5
```
Extend Excellent Taste to the Product Itself
 (Score: 132.0569)
```
```
•       Extend Excellent Taste to the Product Itself        
 (Score: 125.6464)
```
```
•       Extend Excellent Taste to the Product Itself        
 (Score: 125.6464)
```
```
•       Capital & Taste
 (Score: 113.1370)
```
```
Capital & Taste
 (Score: 111.8833)
```