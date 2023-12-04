# Using `systemd`

We can run the server using a `systemd` service. To do so, follow these steps:

1. Copy the `dreamia.service` to `/etc/systemd/system/dreamia.service`;
2. Edit the file (`sudo vim /etc/systemd/system/dreamia.service`) with the correct user and path;
3. Run `sudo chmod 644 /etc/systemd/system/dreamia.service`;
4. Enable the service `systemctl enable dreamia.service`;
5. Run the service `systemctl start dreamia.service`;
6. (optional) To see if it is running check its status `systemctl status dreamia.service` or see the logs `journalctl -u dreamia.service` (add `-f` to follow the file).

See this link for more information https://www.baeldung.com/linux/run-script-on-startup