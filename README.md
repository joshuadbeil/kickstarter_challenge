| Index | Variable                | Description                                                                                                                   |		
|-------|------------------------|-------------------------------------------------------------------------------------------------------------------------------|		
|	0	|	backers_count	|	number of people that backed the project
|	1	|	blurb	|	short description of the project
|	2	|	category	|	category and parent category of the project: id, name, slug, position, parent_id, color, urls
|	3	|	converted_pledged_amount	|	pledged amount converted to USD
|	4	|	country	|	creator's country of origin
|	5	|	created_at	|	date the project was created
|	6	|	creator	|	dictionary with information about the creator: id, name, is_registered, chosen_currency, avatar, urls:web:user (user profile)
|	7	|	currency	|	the projects original currency
|	8	|	currency_symbol	|	symbol for that, (can be dropped because its colinear with currency)
|	9	|	currency_trailing_code	|	???
|	10	|	current_currency	|	???
|	11	|	deadline	|	deadline for when the project needs to be financed
|	12	|	disable_communication	|	maybe if comments are turned off?
|	13	|	friends	|	friends, is_backing, is_starred, permissions appear to be perfectly colinear. CHECK WHAT THATS ABOUT
|	14	|	fx_rate	|	some currency conversion rate between USD and 'currency'
|	15	|	goal	|	funding goal in the currency of 'currency'
|	16	|	id	|	project id
|	17	|	is_backing	|	friends, is_backing, is_starred, permissions appear to be perfectly colinear. CHECK WHAT THATS ABOUT
|	18	|	is_starrable	|	boolean value if project could potentially be starred
|	19	|	is_starred	|	friends, is_backing, is_starred, permissions appear to be perfectly colinear. CHECK WHAT THATS ABOUT
|	20	|	launched_at	|	date the project was launched
|	21	|	location	|	project's location of origin (dictionary needs to be decoded)
|	22	|	name	|	name of the project
|	23	|	permissions	|	friends, is_backing, is_starred, permissions appear to be perfectly colinear. CHECK WHAT THATS ABOUT
|	24	|	photo	|	photo of the project: IS THERE ONLY ONE OR ARE THERE MULTIPLE IN THE DICTIONARY?
|	25	|	pledged	|	pledged amount of money in the original 'currency'
|	26	|	profile	|	dictionary with profile information: project id, state, state_changed_at (how is this different from the other state column? it def. is different) information about the project, pictures of the project
|	27	|	slug	|	slug encoded into string without whitespace and .lower(), probably for url
|	28	|	source_url	|	url to the project page
|	29	|	spotlight	|	whether the project was spotlighted. WHAT'S THAT?
|	30	|	staff_pick	|	whether the project was picked by the staff? picked for what? manually?
|	31	|	state	|	['successful', 'failed', 'live', 'canceled', 'suspended']
|	32	|	state_changed_at	|	last time the state changed
|	33	|	static_usd_rate	|	some currency conversion rate between USD and 'currency'
|	34	|	urls	|	another url to the project and a url to the rewards/pledge levels
|	35	|	usd_pledged	|	pledged amount converted to USD using 'static_usd_rate'
|	36	|	usd_type	|	international', 'domestic', nan

---
## Requirements and Environment

Requirements:
- pyenv with Python: 3.9.8

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.
