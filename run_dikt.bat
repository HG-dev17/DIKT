@echo off
chcp 65001 >nul
REM 激活DIKT的conda环境并运行main.py
REM 使用方法: 在cmd中运行此脚本，或在资源管理器中双击

echo ========================================
echo DIKT 训练脚本
echo ========================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查conda是否可用
where conda >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到conda命令
    echo 请确保已安装Anaconda/Miniconda并添加到PATH
    echo 或者使用Anaconda Prompt运行此脚本
    pause
    exit /b 1
)

REM 初始化conda（Windows cmd需要特殊处理）
if not defined CONDA_DEFAULT_ENV (
    echo 正在初始化conda...
    call conda init cmd.exe >nul 2>&1
)

REM 尝试激活conda环境（使用call确保批处理继续执行）
echo 正在激活conda环境 dikt...
call conda activate dikt

if errorlevel 1 (
    echo.
    echo 错误: 无法激活conda环境 dikt
    echo 请确保已创建conda环境，运行以下命令:
    echo   conda create -n dikt python=3.7
    echo   conda activate dikt
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo conda环境 dikt 已激活
echo 当前Python路径: 
where python
echo.

REM 创建输出目录（如果不存在）
if not exist "output" (
    echo 创建输出目录 output...
    mkdir output
)

REM 检查数据文件是否存在
if not exist "pre_process_data\assist09\0\train_test\train_question.txt" (
    echo.
    echo 警告: 未找到数据文件
    echo 请确保已运行数据预处理脚本 preprocess_data.py
    echo 数据文件应位于: pre_process_data\assist09\{fold}\train_test\
    echo.
    set /p continue="是否继续运行? (Y/N): "
    if /i not "%continue%"=="Y" (
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo 正在运行 main.py...
echo ========================================
echo.

python main.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo 程序运行出错，请检查错误信息
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo 程序运行完成
echo ========================================
pause
