#!/bin/bash
echo "=== TRAINING STATUS CHECK ==="
echo ""

# Check if process is running
if pgrep -f "train_from_human.py" > /dev/null; then
    echo "⏳ TRAINING IS STILL RUNNING"
    ELAPSED=$(ps -p $(pgrep -f train_from_human.py) -o etime= | xargs)
    CPU=$(ps -p $(pgrep -f train_from_human.py) -o %cpu= | xargs)
    echo "   Runtime: $ELAPSED"
    echo "   CPU: $CPU%"
else
    echo "✅ TRAINING IS COMPLETE!"
    echo ""
    echo "Check results:"
    echo "   tail -100 human_training_CORRECT.log"
fi

echo ""
echo "Last 5 lines of log:"
tail -5 human_training_CORRECT.log 2>/dev/null || echo "   (no log)"
echo ""
