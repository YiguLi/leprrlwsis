# Control experiment: send one toast synchronously and print any WinRT exception verbatim.
# Same template and same AUMID as notify-toast.mjs (borrowing PowerShell's own registered AUMID).
# ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less script files as ANSI/GBK.
$ErrorActionPreference = 'Stop'
try {
  [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
  [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

  $aumid = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
  $x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
  $n = $x.GetElementsByTagName('text')
  $n.Item(0).AppendChild($x.CreateTextNode('DSH toast self-test')) | Out-Null
  $n.Item(1).AppendChild($x.CreateTextNode('If you can see this, the toast pipeline works')) | Out-Null

  [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($aumid).Show([Windows.UI.Notifications.ToastNotification]::new($x))
  Write-Host 'RESULT=SHOWN  no exception from CreateToastNotifier/Show'
} catch {
  Write-Host ('RESULT=THREW  ' + $_.Exception.GetType().FullName)
  Write-Host ('MESSAGE=' + $_.Exception.Message)
}

$p = Get-Process -Name ShellExperienceHost -ErrorAction SilentlyContinue
Write-Host ('ShellExperienceHost=' + $(if ($p) { 'running(pid ' + $p.Id + ')' } else { 'NOT RUNNING' }))
