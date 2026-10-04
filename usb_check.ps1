# Lists every USB port on this PC, whether it has USB3 (SuperSpeed) lanes, and the speed each device is running at.
# Read-only: uses the standard Windows USB hub queries. Look for 03e7 (the OAK camera): "Super(5G)" means USB3 works.
$src = @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class UsbPorts {
    [DllImport("setupapi.dll", SetLastError=true)] static extern IntPtr SetupDiGetClassDevs(ref Guid g, IntPtr e, IntPtr h, uint f);
    [DllImport("setupapi.dll", SetLastError=true)] static extern bool SetupDiEnumDeviceInterfaces(IntPtr s, IntPtr d, ref Guid g, uint i, ref SP_DEVICE_INTERFACE_DATA data);
    [DllImport("setupapi.dll", SetLastError=true, CharSet=CharSet.Unicode)] static extern bool SetupDiGetDeviceInterfaceDetail(IntPtr s, ref SP_DEVICE_INTERFACE_DATA d, IntPtr detail, uint size, out uint req, IntPtr info);
    [DllImport("setupapi.dll")] static extern bool SetupDiDestroyDeviceInfoList(IntPtr s);
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] static extern SafeFileHandle CreateFile(string n, uint a, uint s, IntPtr sa, uint c, uint f, IntPtr t);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool DeviceIoControl(SafeFileHandle h, uint code, byte[] inb, int ins, byte[] outb, int outs, out int ret, IntPtr o);
    [StructLayout(LayoutKind.Sequential)] struct SP_DEVICE_INTERFACE_DATA { public int cbSize; public Guid g; public int flags; public IntPtr r; }

    public static List<string> Scan() {
        var lines = new List<string>();
        Guid hubGuid = new Guid("f18a0e88-c30c-11d0-8815-00a0c906bed8");
        IntPtr set = SetupDiGetClassDevs(ref hubGuid, IntPtr.Zero, IntPtr.Zero, 0x12);
        string[] speeds = { "Low(1.5M)", "Full(12M)", "High(480M) = USB2", "Super(5G) = USB3" };
        for (uint i = 0; ; i++) {
            var d = new SP_DEVICE_INTERFACE_DATA(); d.cbSize = Marshal.SizeOf(d);
            if (!SetupDiEnumDeviceInterfaces(set, IntPtr.Zero, ref hubGuid, i, ref d)) break;
            uint req; SetupDiGetDeviceInterfaceDetail(set, ref d, IntPtr.Zero, 0, out req, IntPtr.Zero);
            IntPtr buf = Marshal.AllocHGlobal((int)req);
            Marshal.WriteInt32(buf, IntPtr.Size == 8 ? 8 : 6);
            SetupDiGetDeviceInterfaceDetail(set, ref d, buf, req, out req, IntPtr.Zero);
            string path = Marshal.PtrToStringUni(buf + 4); Marshal.FreeHGlobal(buf);
            using (var h = CreateFile(path, 0x40000000, 2, IntPtr.Zero, 3, 0, IntPtr.Zero)) {
                if (h.IsInvalid) { lines.Add("hub " + i + ": cannot open"); continue; }
                byte[] ni = new byte[76]; int ret;
                if (!DeviceIoControl(h, 0x220408, ni, ni.Length, ni, ni.Length, out ret, IntPtr.Zero)) continue;
                int ports = ni[6];
                bool root = path.ToUpper().Contains("ROOT_HUB");
                lines.Add(string.Format("hub {0} ({1}, {2} ports)", i, root ? "root hub" : "external hub", ports));
                for (int p = 1; p <= ports; p++) {
                    byte[] c = new byte[512]; BitConverter.GetBytes(p).CopyTo(c, 0);
                    if (!DeviceIoControl(h, 0x220448, c, c.Length, c, c.Length, out ret, IntPtr.Zero)) continue;
                    int status = BitConverter.ToInt32(c, 31);
                    byte[] v2 = new byte[16]; BitConverter.GetBytes(p).CopyTo(v2, 0); BitConverter.GetBytes(16).CopyTo(v2, 4); BitConverter.GetBytes(7).CopyTo(v2, 8);
                    bool okV2 = DeviceIoControl(h, 0x22045C, v2, 16, v2, 16, out ret, IntPtr.Zero);
                    int protos = okV2 ? BitConverter.ToInt32(v2, 8) : 0, flags = okV2 ? BitConverter.ToInt32(v2, 12) : 0;
                    string sup = ((protos & 4) != 0 ? "USB3-capable port" : "USB2-only port");
                    if (status == 1) {
                        int vid = BitConverter.ToUInt16(c, 12), pid = BitConverter.ToUInt16(c, 14), spd = c[23];
                        bool ssNow = (flags & 1) != 0, ssCap = (flags & 2) != 0;
                        lines.Add(string.Format("   port {0}: [{1}] device {2:x4}:{3:x4} speed {4}{5}{6}", p, sup, vid, pid,
                            spd < 4 ? speeds[spd] : spd.ToString(), ssNow ? " (operating SuperSpeed)" : "",
                            (ssCap && !ssNow) ? " (device is SuperSpeed-capable but NOT running SuperSpeed)" : ""));
                    } else if ((protos & 4) != 0) {
                        lines.Add(string.Format("   port {0}: [{1}] empty", p, sup));
                    }
                }
            }
        }
        SetupDiDestroyDeviceInfoList(set);
        return lines;
    }
}
"@
Add-Type -TypeDefinition $src -Language CSharp
[UsbPorts]::Scan() | ForEach-Object { $_ }
