# Keeps Windows from sleeping (Modern Standby pauses the WSL GPU processing) while rgbd_process.py runs in WSL.
# On Modern Standby laptops "system required" alone is not enough - the machine still drops into standby once the
# display turns off - so the display is kept on too. Exits on its own once the processing has finished. Closing the
# lid still sleeps the laptop unless the lid action is set to "Do nothing" in the power settings.
#   powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File keep_awake.ps1
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$api = Add-Type -MemberDefinition $sig -Name Power -Namespace KeepAwake -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$ES_DISPLAY_REQUIRED = [uint32]"0x00000002"
$null = $api::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED)
Start-Sleep -Seconds 60  # give the processing time to start
while ($true) {
    $running = wsl.exe -d Ubuntu-22.04 -- bash -lc "pgrep -f '[p]ython -u rgbd_process' >/dev/null && echo yes || echo no"
    if ("$running".Trim() -ne "yes") { break }
    Start-Sleep -Seconds 60
}
$null = $api::SetThreadExecutionState($ES_CONTINUOUS)
