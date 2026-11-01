import json,re,collections,statistics,sys
sys.path.insert(0,'scripts')
from dedup import F,clean,DROPCAT,D
TW=json.load(open(D+'dl/tw_parsed.json',encoding='utf-8'))
twby=collections.defaultdict(list)
for s in TW: twby[s['zh']].append(s)
def twfind(k):
    if k in twby: return twby[k][0]
    for z,v in twby.items():
        if z.startswith(k+'('): return v[0]
    return None
def slug(s):
    s=re.sub(r'\(.*?\)','',s.lower()); return re.sub(r'[^a-z0-9]+','_',s).strip('_')
def r1(x): return round(x,1) if x is not None else None
out=[];ids=set();misses=[]
def add(e):
    i=e['id'];n=2
    while i in ids: i=f"{e['id']}_{n}";n+=1
    e['id']=i;ids.add(i);out.append(e)
def rng(g): return [int(round(g*0.5/5)*5) or 1,int(round(g*1.6/5)*5) or 2]
# curated dishes
for ln in open(D+'scripts/curated_dishes.txt',encoding='utf-8'):
    if ln.startswith('#') or not ln.strip(): continue
    i,en,zh,cat,pd,g,src=ln.rstrip('\n').split('|');g=float(g)
    if src.startswith('@'):
        s=twfind(src[1:])
        if not s: misses.append(('dish',src));continue
        if s['kcal']<=0 and cat!='drink': misses.append(('zero',src));
        e=dict(id=i,en=en,zh=zh,kcal_per_100g=s['kcal'],protein_g=s.get('p'),carb_g=s.get('c'),fat_g=s.get('f'),source=f"TW-FDA {s['code']} {s['zh']}",confidence='db')
    else:
        k,p,c,f=map(float,src[2:].split(','))
        e=dict(id=i,en=en,zh=zh,kcal_per_100g=k,protein_g=p,carb_g=c,fat_g=f,source='estimated from typical recipe composition (no database entry)',confidence='est')
    e.update(category=cat,portion=dict(desc=pd,g=int(g)),portion_range_g=rng(g))
    add(e)
# ingredients
for ln in open(D+'scripts/curated_ingr.txt',encoding='utf-8'):
    if ln.startswith('#') or not ln.strip(): continue
    key,en,zh,cat,pd,g=ln.rstrip('\n').split('|');g=float(g)
    s=twfind(key)
    if not s: misses.append(('ingr',key));continue
    e=dict(id=slug(en),en=en,zh=zh,kcal_per_100g=s['kcal'],protein_g=s.get('p'),carb_g=s.get('c'),fat_g=s.get('f'),category=cat,portion=dict(desc=pd,g=int(g)),portion_range_g=rng(g),source=f"TW-FDA {s['code']} {s['zh']}",confidence='db')
    add(e)
ncur=len(out)
# FNDDS
DROPDESC=re.compile(r'groundhog|venison|deer|moose|squirrel|beaver|raccoon|opossum|\bbear\b|rabbit|ladoo|caribou|elk\b|bison|frog|turtle|alligator|\bbird\b|pigeon|quail|pheasant|goose|ostrich|emu|snail|escargot|tripe|brain|tongue|lungs|spleen|chitterlings|puerto rican|reported separately|latino|\(mcdonalds|burger king|wendy|taco bell|kfc|subway|pizza hut|domino|starbucks|dunkin|nfs\b.*baby|infant|toddler|supplement|human milk|school lunch|^water,|^tap water|dietary|meal replacement|slim|nutritional|ensure|boost|gerber|hawaiian punch|kool|mexican-style coke|\bkit\b',re.I)
g=collections.defaultdict(list)
for o in F:
    if DROPCAT.match(o['cat']) or re.match(r'(Liquor|Not included|Sugars|Butter and|Margarine|Salad dressings|Mayonnaise|Jams|Dips|Mustard|Soy-based cond|Tomato-based cond|Pasta sauces|Cream|Gelatins)',o['cat']) : continue
    if DROPDESC.search(o['desc']): continue
    if o['kcal']<=0 and not re.search('tea|coffee|water|soda|drink|cola|broth',o['cat']+o['desc'],re.I): continue
    g[(o['cat'],clean(o['desc']))].append(o)
STOP=set('with and or in on of the a an sauce style type flavored flavor plain white wheat whole bread bun roll made from home recipe any other all kind'.split())
def toks(k):
    return frozenset(re.sub(r's$','',w) for w in re.findall(r'[a-z]+',k) if w not in STOP)
TH=float(sys.argv[1]) if len(sys.argv)>1 else 0.6
cl=collections.defaultdict(list)
byc=collections.defaultdict(list)
for (cat,key),mem in g.items(): byc[cat].append((key,mem))
g2={}
for cat,items in byc.items():
    seeds=[]
    for key,mem in sorted(items,key=lambda x:len(x[0])):
        t=toks(key);best=None;bj=0
        for sd in seeds:
            j=len(t&sd[1])/max(1,len(t|sd[1]))
            if j>bj: bj=j;best=sd
        if best and bj>=TH: best[2].extend(mem)
        else: seeds.append([key,t,list(mem)])
    for key,t,mem in seeds: g2[(cat,key)]=mem
g=g2
CATMAP=[('juice|fruit drinks|smoothie|soft drink|tea|coffee|water|milk|beer|wine|sport|energy|milk shake|plant-based milk|drink|Liquor','drink'),
('soup|ramen','soup'),('Rice|Fried rice','rice'),('Pasta|noodle|Macaroni','noodle'),('sandwich|Burger|Pizza|Burrito|Taco|Nacho|Egg/breakfast|Turnover','sandwich'),('bread|Bagel|roll|bun|Biscuit|Tortilla|Pancake|waffle','bread'),
('cake|cookie|candy|ice cream|Doughnut|Pudding|Gelatin|Cereal bar|Jam|dessert','dessert'),('chip|Pretzel|Popcorn|cracker|Nuts|snack|Potato chips|Olives','snack'),
('fruit|Apple|Banana|berries|Citrus|Grapes|Mango|Melon|Peach|Pear|Pineapple|Strawberr','fruit'),('vegetable|Broccoli|Cabbage|Carrot|Corn|Lettuce|Onion|Spinach|String beans|Tomato|salad|Coleslaw|potato|Beans|Bean','vegetable'),
('Fish|Shellfish|Seafood','seafood'),('Beef|Pork|Chicken|Turkey|Lamb|Bacon|Sausage|Frankfurter|cold cuts|Ground beef|Liver|Poultry|Meat','meat'),
('dressing|sauce|Mayonnaise|Butter|Margarine|condiment|Dips|Cream|Gravy','sauce'),('Cheese|Yogurt|Eggs|Cottage','dish')]
def cat_of(c,desc):
    if 'mixed dishes' in c or 'Meat mixed' in c or 'dishes' in c or 'Stir-fry' in c or 'Egg rolls' in c: return 'dish'
    for pat,cat in CATMAP:
        if re.search(pat,c,re.I): return cat
    return 'dish'
def pickport(o):
    P=o['ports']
    if not P: return None
    pref=re.compile(r'\b(sandwich|burger|slice|piece|bowl|serving|cup|roll|taco|burrito|bar|cookie|patty|link|fillet|egg|pancake|waffle|muffin|donut|doughnut|can|bottle|container|medium|small|large|leg|breast|wing|drumstick|fruit|pizza)\b',re.I)
    for d,gw,sq in P:
        if pref.search(d) and gw>=30 and not re.search(r'tablespoon|teaspoon|packet|cubic|fl oz|ounce\b|oz\b',d): return d,gw
    for d,gw,sq in P:
        if gw>=30 and not re.search(r'tablespoon|teaspoon|fl oz',d): return d,gw
    d,gw,sq=max(P,key=lambda x:x[1]); return d,gw
def nice(c,desc):
    segs=[s.strip() for s in c.split(',')]
    if len(segs)==2 and segs[1].startswith('with'): return f"{segs[0]} {segs[1]}"
    if len(segs)==2 and ' ' not in segs[0] and 'with' not in segs[0] and segs[0] not in('soup','cereal','pie','cake','cookie'):
        return f"{segs[1]} {segs[0]}"
    if len(segs)==2 and segs[0] in('soup','cereal','pie','cake','cookie'): return f"{segs[1]} {segs[0]}" if segs[0]!='soup' else f"{segs[1]} soup"
    return ' '.join(segs)

COOK=set('fried baked broiled grilled roasted boiled steamed stewed raw cooked smoked canned dried pickled scrambled poached breaded battered barbecued sauteed braised toasted frozen sweetened'.split())
SUF=set('breast wing wings thigh drumstick leg legs ribs rib chop chops steak roast belly loin ground patty fillet liver sausage nuggets strips tenders slices sticks balls rings salad soup stew sandwich'.split())
NOISEP=re.compile(r'(NFS|^NS|NS as to|not (further )?specified|^fat not added|fat added|^no sauce|^from |^skin|^coated|^lean and fat|^lean only|^separable|^made with|^cooked$|^ready|^low|^reduced|^regular|^nonfat|^fat free|^unsweetened|^without|^sodium|^canned$|^drained|^all types|^any type|^type of|^flavors? other|^plain$|^trimmed|^untrimmed|^excludes|^excluding|, fat|^not |^as to|^prepared|^fortified|^enriched|^imitation|^light|^dry$)',re.I)
def nice2(desc):
    parts=[x.strip() for x in desc.split(',')]
    parts=[x for x in parts if x and not NOISEP.search(x)]
    parts=[re.sub(r'\s*\(.*?\)','',x).strip() for x in parts]
    parts=[x for x in parts if x]
    if not parts: return desc.lower()
    head=parts[0].lower();mods=[m.lower() for m in parts[1:3]]
    cook=[m for m in mods if m.split()[0] in COOK]
    oth=[m for m in mods if m not in cook]
    if head.startswith(('pizza','sandwich')) or ' with ' in head or ' and ' in head or len(head.split())>2:
        base=head+(' '+' '.join(oth) if oth and oth[0].startswith('with') else '')
        return ' '.join(cook+[base])
    pre=[];post=[]
    for m in oth:
        (post if m.split()[0] in SUF or m.startswith('with') else pre).append(m)
    return ' '.join(cook+pre+[head]+post).strip()
for (cat,key),mem in g.items():
    mem=sorted(mem,key=lambda o:o['kcal'])
    nfs=[o for o in mem if 'NFS' in o['desc'] or 'NS as to' in o['desc']]
    rep=nfs[0] if nfs else mem[len(mem)//2]
    pp=pickport(rep) or next((pickport(o) for o in mem if o['ports']),None)
    est=False
    if not pp:
        est=True; pp=('1 serving (est.)',150 if 'drink' not in cat_of(cat,key) else 240)
    gw=pp[1]
    en=re.sub(r'\s+',' ',nice2(rep['desc'])).strip()
    e=dict(id='us_'+slug(en),en=en,zh=None,kcal_per_100g=round(rep['kcal']),protein_g=r1(rep.get('p')),carb_g=r1(rep.get('c')),fat_g=r1(rep.get('f')),category=cat_of(cat,key),
      portion=dict(desc=pp[0]+(' (portion est.)' if est else ''),g=int(round(gw))),portion_range_g=rng(gw),source=f"FNDDS 2021-2023 #{rep['code']} ({rep['desc'][:70]}); {len(mem)} variant(s) merged",confidence='db',_wweia=cat)
    add(e)
print('curated',ncur,'fndds',len(out)-ncur,'misses',misses,file=sys.stderr)
pass
import pickle
M={}
for (cat,key),mem in g.items():
    mm=sorted(mem,key=lambda o:o['kcal']); nfs=[o for o in mm if 'NFS' in o['desc'] or 'NS as to' in o['desc']]
    rep=nfs[0] if nfs else mm[len(mm)//2]
    M[int(rep['code'])]=[(int(o['code']),o['desc']) for o in mem]
json.dump(M,open(D+'../prior/members.json','w'))
print(len(M),file=sys.stderr)
