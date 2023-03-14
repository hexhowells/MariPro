# Mari/Pro
Learning to play Super Mario Bros with AI

Uses [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) for emulating the game.

## Notable RAM Addresses
| Addr | Description |
| ---- | ----------- |
| 0x071A | Current Screen |
| 0x071C | Screen offset |
| 0x0500-0x069F | Tilemap (32 x 13 circular buffer) |
| 0x04AC-0x04AF | Player hitbox (x1, y1, x2, y2) |
| 0x04B0-0x04C3 | Enemy hitboxes (x1, y1, x2, y2) x5 |
| 0x000F-0x0013 | Enemy loaded (0=No, 1=Yes) x5 |
