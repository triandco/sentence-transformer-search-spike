# Introduction
This project attempts to test a couple of algorithms as a means to generate search indexes for paragraphs inside documents as well as large documents.
* [sentence-transformers](https://www.sbert.net)
* document-embeddings using sentence embeddings (In this repository)
* [rank_bm25](https://pypi.org/project/rank-bm25/)

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
pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
The --no-cache-dir flag was there to address [a known issue](https://stackoverflow.com/questions/64850321/windows-keeps-crashing-when-trying-to-install-pytorch-via-pip) with this command freezing computer. You might not need it if you have not install pytorch before.

3. Install sentence-transformer
```powershell
pip install sentence-transformer
```

# Outcome

## Content search in Sam Harris podcast
### Query 1 - ✅ pass 
Sam talks about a scale of suffering to flourishing. His guest has seems to evaluate things from suffering to zero, Sam Harris seems to evaluate things from suffering to zero to flourishing.
* doc[0] (Score: 46.2354)
* doc[2] (Score: 31.1247)
* doc[1] (Score: 22.5042)

### Query 2 - ✅ pass 
His guest has seems to evaluate things from suffering to zero, Sam Harris seems to evaluate things from suffering to zero to flourishing.
* doc[0](Score: 38.9579)
* doc[1](Score: 35.4988)
* doc[2] (Score: 26.6808)

### Query 3 - ❌ fail 
Oh yeah, this reminds me of some Sammy podcast where he speaks to someone who has the view that life is a scale like: -1 —— 0 as in there’s suffering or there’s not. Where as Sammy was viewing more like: -1 —— 0 —— +1 where there’s flourishing to be had.
* doc[1](Score: 52.1846)
* doc[0] (Score: 24.9433)
* doc[2] (Score: 24.5990)

### Query 4 - ✅ pass 
Suffering scale
* doc[0] (Score: 58.3829)
* doc[2] (Score: 24.4793)
* doc[1] (Score: 2.0260)

### Query 5 - ✅ pass
Sam Harris suffering scale
* doc[0](Score: 31.0392)
* doc[1](Score: 17.4912)
* doc[2] (Score: 17.3035)

### Query 6 - ✅ pass
suffering flourishing scale
* doc[0] (Score: 51.7766)
* doc[1] (Score: 45.4267)
* doc[2] (Score: 16.7268)

### Query 7 - ✅ pass
guest evaluates things from suffering to zero, Sam evaluates things from suffering to zero to flourishing
* doc[0] (Score: 55.6910)
* doc[1] (Score: 25.8012)
* doc[2] (Score: 23.3835)

### Query 8 - ✅ pass
suffering to zero to flourishing
* doc[0] (Score: 95.5005)
* doc[2] (Score: 39.1806)
* doc[1] (Score: 14.7422)

## Paragraph search in Capital and Taste podcast
Document is split by new line characters which indicate a new paragraph. Each paragraph receive its own embedding which is used to compare against embedding of a query 

### Query for "3 principles" - ❌ fail 
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
### Query for "3 approaches"  - ✅ pass
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


### Query for 'younger generation' - ✅ pass
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


### Query for 'extending its taste' - ✅ pass
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