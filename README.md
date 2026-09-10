# SDR Experiments

These are scripts for various experiments I've run on software defined radio.

`frs_scan` includes a couple of tools for listening to voice traffic on FRS/GMRS bands, which is where a lot of walkie-talkies and handheld radios tend to operate. 

Included are scripts for analog voice traffic, which listen on a wide 10MHz band and detect bursts of transmissions in real time, writing only transmissions to disk as IQ captures. These are run through a simple classifier which guesses whether they are voice or not, and if so, writes the audio waveform to a separate folder. `watch.py` will run the full analog voice pipeline. To run with a HackRF radio, try this command:

```
hackrf_transfer -r - -f 465000000 -s 10000000 -l 24 -g 20 -a 0 | python -m frs_scan.watch -f 465e6 -s 10e6 --save-dir hits --wav-dir audio --keep voice
```
Adjust the frequencies as you like, this will listen on 460-470MHz.

Also included are tools for decoding digital (DMR) voice transmissions, which are a little trickier. These have more external dependencies -- specifically, mbelib and dsd-fme -- which have to be compiled from source on almost all distros because they implement a proprietary vocoder. For digital transmissions, first identify a channel of interest (either by going through transmissions from a wideband capture which were marked digital, or by visually screening in a program like SDR++) as you will be limited to listening on a single channel per terminal session. Then, run 

```
hackrf_transfer -r - -f 461875000 -s 4800000 -l 24 -g 20 -a 0 | python -m frs_scan.camp -f 461.875e6 -s 4.8e6 -c 462.3753e6 --tcp
```
in one terminal (note: adjust frequencies to your specific channel), and 

```
dsd-fme -fs -i tcp -o pulse -P -7 decoded
```
in another. You should hear DMR voice transmissions as they appear, and you will see them being written to your audio directory as well.