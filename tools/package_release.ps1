# Builds the portable Windows zip from a finished Release build.
#
#   pwsh tools/package_release.ps1 -BuildDir build_release -QtDir C:/Qt/6.10.1/msvc2022_64 `
#        -OrtSourceDir <ONNX Runtime 1.30 source checkout> [-OutDir dist]
#
# The build is expected to be configured the way the release is:
#   -DDATAFORM_ENABLE_TRAINING=ON -DDATAFORM_ENABLE_LLAMACPP=OFF -DDATAFORM_ENABLE_VOICE=OFF
#
# Produces <OutDir>/Dataform-<version>-win-x64.zip and a .sha256 file next to it.
# The version is read from CMakeLists.txt. Visual C++ runtime DLLs are not
# bundled (README_FIRST.txt points at the redistributable), and neither are
# CUDA or cuDNN.

param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$QtDir,
    [Parameter(Mandatory = $true)][string]$OrtSourceDir,
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent

$cmake = Get-Content (Join-Path $repo "CMakeLists.txt") -Raw
if ($cmake -notmatch 'project\(Dataform VERSION ([0-9.]+)') { throw "No version in CMakeLists.txt" }
$version = $Matches[1]
$name = "Dataform-$version-win-x64"

$OutDir = [IO.Path]::GetFullPath((Join-Path $repo $OutDir))
$stage = Join-Path $OutDir $name
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null

$bin = Join-Path ([IO.Path]::GetFullPath((Join-Path $repo $BuildDir))) "Release"
Copy-Item (Join-Path $bin "Dataform.exe"), (Join-Path $bin "onnxruntime.dll") $stage

# Qt runtime: release DLLs only, no translations, no software OpenGL, no
# Windows-SDK D3D compiler (Windows has its own), no compiler runtime.
& (Join-Path $QtDir "bin/windeployqt.exe") --release --qmldir (Join-Path $repo "qml") `
    --no-translations --no-system-d3d-compiler --no-opengl-sw --no-compiler-runtime `
    (Join-Path $stage "Dataform.exe")
if ($LASTEXITCODE -ne 0) { throw "windeployqt failed" }

# Trim what windeployqt adds but the app does not use: the QML debugger
# plugins, SQL drivers other than SQLite (they need client libraries that are
# not here), and the copy of Windows' own icuuc.dll.
Remove-Item (Join-Path $stage "qmltooling") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $stage "icuuc.dll") -Force -ErrorAction SilentlyContinue
Get-ChildItem (Join-Path $stage "sqldrivers") -Filter *.dll |
    Where-Object { $_.Name -ne "qsqlite.dll" } | Remove-Item -Force

# windeployqt also drops in Qt6Quick3DUtils.dll, which nothing here loads, and
# empty folders for QML modules that are not deployed. Qt Quick 3D is GPL-only.
Remove-Item (Join-Path $stage "Qt6Quick3DUtils.dll") -Force -ErrorAction SilentlyContinue
Get-ChildItem $stage -Directory -Recurse | Sort-Object { $_.FullName.Length } -Descending |
    Where-Object { -not (Get-ChildItem $_.FullName -Force) } | Remove-Item -Force

# The package is distributed under DATAFORM's own license, so it must not carry
# any Qt module that The Qt Company offers only under the GPL or commercially.
$gplOnly = 'Qt6(Charts|DataVisualization|Graphs|Quick3D|VirtualKeyboard|NetworkAuth|QuickTimeline|Lottie|Coap|Mqtt|Opcua|HttpServer|SpatialAudio|ShaderTools|StateMachine|Protobuf|Grpc|LocationQuick|Location)'
$gplQml = '^(QtCharts|QtDataVisualization|QtGraphs|QtQuick3D|QtVirtualKeyboard|QtNetworkAuth)$|QtQuick[\\/](Timeline|VirtualKeyboard)'
$bad = @(Get-ChildItem $stage -Recurse -File | Where-Object { $_.Name -match $gplOnly })
$bad += @(Get-ChildItem (Join-Path $stage "qml") -Recurse -Directory |
    Where-Object { $_.FullName.Substring((Join-Path $stage "qml").Length + 1) -match $gplQml })
if ($bad.Count -gt 0) {
    $bad | ForEach-Object { Write-Host "GPL-only Qt content: $($_.FullName)" }
    throw "The package contains GPL-only Qt modules"
}

# The offscreen platform plugin lets the app start without a desktop session,
# which is how the release is smoke-tested.
Copy-Item (Join-Path $QtDir "plugins/platforms/qoffscreen.dll") (Join-Path $stage "platforms")

# App files and documents.
Copy-Item (Join-Path $repo "dataform.ico"), (Join-Path $repo "LICENSE") $stage
Copy-Item (Join-Path $repo "packaging/THIRD_PARTY_NOTICES.txt") $stage
(Get-Content (Join-Path $repo "packaging/README_FIRST.txt") -Raw).Replace("@VERSION@", $version) |
    Set-Content (Join-Path $stage "README_FIRST.txt") -NoNewline -Encoding utf8

# Full license texts for everything named in THIRD_PARTY_NOTICES.txt. Qt keeps
# them, by SPDX id, in the LICENSES folders of its source modules.
$licenses = Join-Path $stage "licenses"
New-Item -ItemType Directory -Force $licenses | Out-Null
$qtSrc = Join-Path (Split-Path $QtDir -Parent) "Src"
$spdx = @("LGPL-3.0-only", "GPL-3.0-only", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
          "CC0-1.0", "FTL", "GPL-2.0-only", "HPND-sell-variant", "IJG", "Imlib2", "Libpng",
          "libpng-2.0", "LicenseRef-BSD-3-Clause-with-PCRE2-Binary-Like-Packages-Exception",
          "LicenseRef-ICC-License", "LicenseRef-Lcs-Telegraphics", "LicenseRef-SHA1-Public-Domain",
          "MIT", "MIT-open-group", "MPL-2.0", "Unicode-3.0", "Zlib", "blessing")
foreach ($id in $spdx) {
    $src = @("qtbase", "qtdeclarative", "qtsvg") |
        ForEach-Object { Join-Path $qtSrc "$_/LICENSES/$id.txt" } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $src) { throw "License text $id.txt not found under $qtSrc" }
    Copy-Item $src $licenses
}
$ortLic = Join-Path $licenses "onnxruntime"
New-Item -ItemType Directory -Force $ortLic | Out-Null
Copy-Item (Join-Path $OrtSourceDir "LICENSE"), (Join-Path $OrtSourceDir "ThirdPartyNotices.txt") $ortLic

# Zip with the folder at the top level, then record its SHA-256.
$zip = Join-Path $OutDir "$name.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Set-Content "$zip.sha256" -Encoding ascii
Write-Host "Wrote $zip"
Write-Host "SHA-256 $hash"
