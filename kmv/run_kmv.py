"""
KMV 모델 실행 스크립트
사용법: python run_kmv.py [데이터파일명]
"""

import sys
import os
from kmv_module import KMVModel


def main():
    # 폴더 경로 설정
    data_dir = 'data'
    results_dir = 'results'
    
    # 폴더 생성 (없는 경우)
    for dir_path in [data_dir, results_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"'{dir_path}' 폴더를 생성했습니다.")
    
    # 기본 파일명 설정
    default_files = ['kp_kmv.parquet', 'kd_kmv.parquet']
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        input_file = os.path.join(data_dir, filename)
    else:
        # 기본 파일 중 존재하는 것 찾기
        input_file = None
        for file in default_files:
            full_path = os.path.join(data_dir, file)
            if os.path.exists(full_path):
                input_file = full_path
                break
        
        if not input_file:
            print(f"\n오류: '{data_dir}' 폴더에서 데이터 파일을 찾을 수 없습니다.")
            print(f"필요한 파일: {default_files}")
            print(f"\n사용법:")
            print(f"  python run_kmv.py              # 자동으로 파일 찾기")
            print(f"  python run_kmv.py kd_kmv.parquet  # 특정 파일 지정")
            return
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"오류: 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # 출력 파일명 생성 (results 폴더에 저장)
    base_name = os.path.basename(input_file)
    if 'kp' in base_name.lower():
        output_file = os.path.join(results_dir, 'kmv_pd_kp.parquet')
    elif 'kd' in base_name.lower():
        output_file = os.path.join(results_dir, 'kmv_pd_kd.parquet')
    else:
        output_name = base_name.replace('.parquet', '_pd.parquet')
        output_file = os.path.join(results_dir, output_name)
    
    print(f"\n입력 파일: {input_file}")
    print(f"출력 폴더: {results_dir}/")
    print(f"출력 파일: {os.path.basename(output_file)}")
    
    # KMV 모델 실행
    kmv_model = KMVModel()
    result = kmv_model.run(data_path=input_file, output_path=output_file)
    
    print(f"\n✅ 처리 완료!")
    print(f"   - 결과 행 수: {len(result):,}")
    print(f"   - 저장 위치: {results_dir}/")


if __name__ == "__main__":
    main()