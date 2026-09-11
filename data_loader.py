import kagglehub
import shutil
from pathlib import Path

# Kaggle 데이터 다운로드
path = kagglehub.dataset_download(
    "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
)

target = Path("./data")
target.mkdir(exist_ok=True)

for file in Path(path).iterdir():
    shutil.copy2(file, target / file.name)

print(f"저장 완료: {target}")