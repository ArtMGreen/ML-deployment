from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import os

from datetime import timedelta, datetime


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 2),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'grade_prediction_pipeline',
    default_args=default_args,
    description='Imputes and splits the data, trains the model and deploys it',
    schedule_interval=timedelta(minutes=5),
    dagrun_timeout=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1
)    # schedule_interval=timedelta(days=1)

def load_dataset_from_csv(**kwargs):
    raw_data = pd.read_csv("/home/artmgreen/DataspellProjects/ML-deployment/data/raw/grades.csv", sep=";")
    kwargs['ti'].xcom_push(key='raw_data', value=raw_data)
    print("Read data/raw/grades.csv and passed the dataframe")


def impute_dataset(**kwargs):
    raw_data = kwargs['ti'].xcom_pull(key='raw_data', task_ids='load_dataset_from_csv')
    imputed = raw_data.replace({'-': 0}).astype('float64')
    kwargs['ti'].xcom_push(key='imputed', value=imputed)
    print("Imputed data")


def split_dataset(**kwargs):
    imputed = kwargs['ti'].xcom_pull(key='imputed', task_ids='impute_dataset')
    train_data = imputed.sample(frac=0.8)
    test_data = imputed.drop(train_data.index)
    train_data.to_csv("/home/artmgreen/DataspellProjects/ML-deployment/data/processed/train.csv", index=False)
    test_data.to_csv("/home/artmgreen/DataspellProjects/ML-deployment/data/processed/test.csv", index=False)
    print("Saved into data/processed/train.csv and data/processed/test.csv")


loading = PythonOperator(
    task_id='load_dataset_from_csv',
    python_callable=load_dataset_from_csv,
    dag=dag,
    provide_context=True
)

imputing = PythonOperator(
    task_id='impute_dataset',
    python_callable=impute_dataset,
    dag=dag,
    provide_context=True
)

splitting = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset,
    dag=dag,
    provide_context=True
)

training = BashOperator(
    task_id='train_model',
    bash_command="cd /home/artmgreen/DataspellProjects/ML-deployment && python src_code/models/model_train.py",
    dag=dag
)

deployment = BashOperator(
    task_id='deploy_model',
    bash_command="cd /home/artmgreen/DataspellProjects/ML-deployment && docker compose -f src_code/deployment/docker-compose.yml up",
    # or you can force rebuilding by including --build parameter at the end
    dag=dag
)

loading >> imputing >> splitting >> training >> deployment