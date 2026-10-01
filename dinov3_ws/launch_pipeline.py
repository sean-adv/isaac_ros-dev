"""由 run.sh 啟動兩個節點，並統一處理退出與訊號。"""
import os
import shlex
from pathlib import Path
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def main():
    workspace = Path(__file__).resolve().parent
    params = Path(os.environ.get(
        'RGBD_PARAMS_FILE', workspace / 'src/rgbd_object_proposals/config/proposals.yaml'
    )).expanduser().resolve()
    if not params.is_file():
        print(f'找不到 RGB-D 參數檔：{params}', file=sys.stderr)
        return 2
    proposal = Node(
        package='rgbd_object_proposals', executable='rgbd_object_proposal_node',
        name='rgbd_object_proposal_node', parameters=[str(params)],
        prefix=shlex.join([sys.executable, '-u']), output='screen',
    )
    # 明確使用 venv Python，避免 colcon 產生的入口 shebang 指向系統 Python。
    recognition = ExecuteProcess(
        cmd=[sys.executable, '-u', '-m', 'dinov3_cases.node',
             '--ros-args', '-p', 'use_proposals:=true',
             '-p', 'proposal_image_topic:=/rgbd_proposals/rgb',
             '-p', 'proposals_topic:=/rgbd_proposals/detections',
             '-p', f'cases_dir:={workspace.parent / "dinov3_cases_data"}',
             *sys.argv[1:]],
        name='dinov3_cases', output='screen',
    )
    failures = []

    def stop_pipeline(event, context):
        if event.returncode:
            failures.append(event.returncode)
        if not context.is_shutdown:
            return [EmitEvent(event=Shutdown(reason='流程中一個節點已退出，關閉另一個節點'))]
        return []

    service = LaunchService(noninteractive=not sys.stdin.isatty())
    service.include_launch_description(LaunchDescription([
        RegisterEventHandler(OnProcessExit(target_action=proposal, on_exit=stop_pipeline)),
        RegisterEventHandler(OnProcessExit(target_action=recognition, on_exit=stop_pipeline)),
        proposal, recognition,
    ]))
    result = service.run()
    return result or (1 if failures else 0)


if __name__ == '__main__':
    raise SystemExit(main())
