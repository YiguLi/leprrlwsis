# End-to-end retest of the toast path, replicating notify-toast.mjs exactly:
# same AUMID, same UTF-16LE -EncodedCommand transfer, same UTF-8-base64 payload,
# same ToastText02 template. Sends three toasts with gaps so they are individually visible.
# ASCII-only source: Windows PowerShell 5.1 reads BOM-less scripts as ANSI/GBK.
$ErrorActionPreference = 'Stop'
$aumid = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
$log = Join-Path $env:TEMP 'dsh-notify-toast.log'

[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$items = @(
  @{ Title = 'DSH'; Body = [char]0x901A + [char]0x77E5 + [char]0x5DF2 + [char]0x542F + [char]0x7528 },                      # tong zhi yi qi yong
  @{ Title = 'DSH'; Body = [char]0x7B49 + [char]0x5F85 + [char]0x56DE + [char]0x7B54 + [char]0xFF1A + 'retest' },           # deng dai hui da :
  @{ Title = 'DSH'; Body = [char]0x56DE + [char]0x5408 + [char]0x7ED3 + [char]0x675F + ' - retest' }                        # hui he jie shu
)

foreach ($it in $items) {
  try {
    $x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $n = $x.GetElementsByTagName('text')
    $n.Item(0).AppendChild($x.CreateTextNode($it.Title)) | Out-Null
    $n.Item(1).AppendChild($x.CreateTextNode($it.Body)) | Out-Null
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($aumid).Show([Windows.UI.Notifications.ToastNotification]::new($x))
    $line = "SENT ok title=$($it.Title) body=$($it.Body)"
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
  } catch {
    $line = 'THREW ' + $_.Exception.GetType().FullName + ' :: ' + $_.Exception.Message
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
  }
  Start-Sleep -Milliseconds 1800
}

Write-Host ''
Write-Host '=== plugin log tail ==='
if (Test-Path $log) { Get-Content $log -Tail 8 }
$d = 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Notifications\Data'
Write-Host ('=== Notifications\Data 记录数 = ' + $(if (Test-Path $d) { (Get-ChildItem $d | Measure-Object).Count } else { 'key absent' }) + ' ===')
