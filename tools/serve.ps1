# Fallback local web server for Windows PCs without Python (uses built-in .NET).
# Run via start-demo.bat, or:  powershell -ExecutionPolicy Bypass -File tools\serve.ps1
param([int]$Port = 8000)

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$types = @{
  '.html' = 'text/html; charset=utf-8'; '.js' = 'text/javascript; charset=utf-8'; '.css' = 'text/css; charset=utf-8'
  '.json' = 'application/json'; '.bin' = 'application/octet-stream'; '.md' = 'text/plain; charset=utf-8'
  '.jpg' = 'image/jpeg'; '.jpeg' = 'image/jpeg'; '.png' = 'image/png'; '.svg' = 'image/svg+xml'; '.ico' = 'image/x-icon'
}

$listener = New-Object System.Net.HttpListener
$started = $false
for ($p = $Port; $p -lt $Port + 20; $p++) {
  try {
    $listener = New-Object System.Net.HttpListener
    $listener.Prefixes.Add("http://localhost:$p/")
    $listener.Start()
    $Port = $p; $started = $true; break
  } catch { }
}
if (-not $started) { Write-Host 'Could not start a local web server.'; Read-Host 'Press Enter to close'; exit 1 }

$url = "http://localhost:$Port/"
Write-Host ""
Write-Host "  Cat or Alligator demo is running at $url"
Write-Host "  Leave this window open while you use the demo. Close it to stop."
Write-Host ""
Start-Process $url

while ($listener.IsListening) {
  $ctx = $listener.GetContext()
  try {
    $rel = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath).TrimStart('/')
    if ($rel -eq '') { $rel = 'index.html' }
    $full = [IO.Path]::GetFullPath((Join-Path $root $rel))
    if ($full.StartsWith($root) -and (Test-Path -LiteralPath $full -PathType Leaf)) {
      $bytes = [IO.File]::ReadAllBytes($full)
      $ext = [IO.Path]::GetExtension($full).ToLower()
      if ($types.ContainsKey($ext)) { $ctx.Response.ContentType = $types[$ext] } else { $ctx.Response.ContentType = 'application/octet-stream' }
      $ctx.Response.AddHeader('Cache-Control', 'no-cache')
      $ctx.Response.ContentLength64 = $bytes.Length
      $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    } else {
      $ctx.Response.StatusCode = 404
    }
  } catch { } finally { $ctx.Response.Close() }
}
