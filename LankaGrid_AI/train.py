import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES=['lag_1','lag_2','lag_3','lag_7','lag_14','lag_28','mean_7','mean_14','std_7','dayofweek','month','dayofyear','weekend']

def load_data(path, date_col, target_col):
    df=pd.read_csv(path)
    if date_col not in df: raise ValueError(f'Date column not found. Available: {list(df.columns)}')
    if target_col in df:
        df=df[[date_col,target_col]].rename(columns={date_col:'date',target_col:'energy_gwh'})
        df['date']=pd.to_datetime(df['date'],errors='coerce')
        df['energy_gwh']=pd.to_numeric(df['energy_gwh'],errors='coerce')
    else:
        df=df.rename(columns={date_col:'date'})
        df['date']=pd.to_datetime(df['date'],errors='coerce')
        numeric_columns=[c for c in df.columns if c!='date']
        values=df[numeric_columns].replace('Data N/A',np.nan).apply(pd.to_numeric,errors='coerce')
        df['energy_gwh']=values.sum(axis=1).where(values.notna().any(axis=1))
    df=df.dropna().sort_values('date')
    if df.date.duplicated().any(): raise ValueError('Duplicate dates: aggregate to one daily total first.')
    if (df.energy_gwh<=0).any(): raise ValueError('Nonpositive generation detected; check data and units.')
    df=df.set_index('date')
    if len(df)<150: raise ValueError('Need at least 150 consecutive daily observations.')
    return df

def features(series):
    d=pd.DataFrame({'energy_gwh':series})
    for lag in [1,2,3,7,14,28]:d[f'lag_{lag}']=series.shift(lag)
    for w in [7,14]:
        d[f'mean_{w}']=series.shift(1).rolling(w).mean()
    d['std_7']=series.shift(1).rolling(7).std()
    d['dayofweek']=d.index.dayofweek;d['month']=d.index.month;d['dayofyear']=d.index.dayofyear
    d['weekend']=(d.index.dayofweek>=5).astype(int)
    return d.dropna()

def build_models():
    return {
      'Decision Tree':DecisionTreeRegressor(max_depth=7,min_samples_leaf=4,random_state=42),
      'Bagging':BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8,min_samples_leaf=3),n_estimators=70,random_state=42,n_jobs=1),
      'Random Forest':RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=42,n_jobs=1),
      'Gradient Boosting':GradientBoostingRegressor(n_estimators=120,learning_rate=.05,max_depth=3,random_state=42),
      'Histogram Boosting':HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,random_state=42),
      'AdaBoost':AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4,random_state=42),n_estimators=100,learning_rate=.05,random_state=42),
      'XGBoost':XGBRegressor(n_estimators=200,learning_rate=.05,max_depth=4,min_child_weight=3,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=42,n_jobs=1),
      'Stacking':StackingRegressor(estimators=[('rf',RandomForestRegressor(n_estimators=70,min_samples_leaf=3,random_state=42,n_jobs=1)),('gb',GradientBoostingRegressor(n_estimators=90,random_state=42))],final_estimator=Ridge(alpha=10),cv=3,n_jobs=1)
    }

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw=load_data(args.csv,args.date_col,args.target_col)
    d=features(raw.energy_gwh)
    split=int(len(d)*.8)
    train,test=d.iloc[:split],d.iloc[split:]
    Xtr,ytr=train[FEATURES],train.energy_gwh
    Xte,yte=test[FEATURES],test.energy_gwh
    models=build_models(); preds={}; results=[]
    for name,model in models.items():
        model.fit(Xtr,ytr)
        preds[name]=model.predict(Xte)
        joblib.dump(model,out/(name.lower().replace(' ','_')+'.joblib'))
    # Fixed hybrid weights are illustrative, not tuned against the held-out test set.
    preds['Hybrid RF + GB']=.5*preds['Random Forest']+.5*preds['Gradient Boosting']
    preds['Hybrid XGB + GB']=.5*preds['XGBoost']+.5*preds['Gradient Boosting']
    for name,p in preds.items():
        results.append({'model':name,'MAE':float(mean_absolute_error(yte,p)),'RMSE':float(np.sqrt(mean_squared_error(yte,p))),'R2':float(r2_score(yte,p))})
    pd.DataFrame(results).sort_values('MAE').to_csv(out/'metrics.csv',index=False)
    pframe=pd.DataFrame({'date':test.index,'actual':yte.to_numpy()})
    for name,p in preds.items():pframe[name]=p
    pframe.to_csv(out/'predictions.csv',index=False)
    raw.reset_index().to_csv(out/'cleaned_daily.csv',index=False)
    joblib.dump({'history':raw.energy_gwh,'features':FEATURES},out/'history.joblib')
    source_path = Path(args.csv).resolve()
    project_root = Path(__file__).resolve().parent
    try:
        source_csv = str(source_path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        source_csv = str(source_path)
    (out/'metadata.json').write_text(json.dumps({'source_csv':source_csv,'date_column':args.date_col,'target_column':args.target_col,'train_rows':len(train),'test_rows':len(test),'test_start':str(test.index.min().date()),'test_end':str(test.index.max().date()),'target':'daily total generation (GWh); demand proxy','method':'chronological 80/20 split; one-day-ahead lag features; AdaBoost, XGBoost, stacking, and fixed-weight hybrids'},indent=2))
    print(pd.DataFrame(results).sort_values('MAE').to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--date-col',required=True);p.add_argument('--target-col',required=True);p.add_argument('--output',default='outputs');main(p.parse_args())
