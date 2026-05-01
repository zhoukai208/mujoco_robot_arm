#!/bin/bash
# 快捷运行脚本

# 从主仓库根目录运行
cd "$(dirname "$0")/.."

echo "=================================================="
echo "📷 Eye-to-Hand 相机标定工具包"
echo "=================================================="
echo ""
echo "🔧 激活环境..."
source .venv/bin/activate
echo ""

# 运行标定
cd camera_calibration
python -c "
import sys
sys.path.insert(0, 'scripts')
from camera_calibration_simple import main
main()
"
