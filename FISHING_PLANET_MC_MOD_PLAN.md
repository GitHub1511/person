# Fishing Planet → Minecraft 1.21.1 Fabric Mod Plan

## Executive Summary
Create a Fabric mod for Minecraft 1.21.1 that brings Fishing Planet's fish, rods, reels, lures, and tackle into Minecraft, with:
- All water bodies spawning Fishing Planet fish species
- Full item registry accessible via `/give` and creative tabs
- Configurable fish spawn rates/frequency
- Compatibility with Lunar Client (Fabric 0.19.5), Bliss shaders, Optimum Realism resource pack

---

## Environment Verification (Complete)

| Component | Status | Location |
|-----------|--------|----------|
| Minecraft 1.21.1 | ✅ Installed | `%APPDATA%\.minecraft\versions\1.21.1` |
| Fabric Loader 0.19.5 | ✅ Installed (via Lunar) | `%APPDATA%\.minecraft\.ichor\ichormodule-libs\net_fabricmc_fabric_loader_0_19_5-lunar-0.1.5.jar` |
| Bliss Shader v2.1.2 | ✅ Installed | `%APPDATA%\.minecraft\shaderpacks\Bliss_v2.1.2_(Chocapic13_Shaders_edit).zip` |
| Optimum Realism R4.1.2 64x | ✅ Installed | `%APPDATA%\.minecraft\resourcepacks\Optimum Realism R4.1.2 64x.zip` |
| Lunar Client | ✅ Installed | `%LOCALAPPDATA%\Programs\Lunar Client` |
| Fishing Planet (Unity/IL2CPP) | ✅ Installed | `C:\Program Files (x86)\Steam\steamapps\common\Fishing Planet` |
| Universal Modder repo | ✅ Accessible | `https://github.com/rehan-remade/universal-modder` |

---

## Phase 0: Toolchain & Safety Setup (M0)

### 0.1 Install Build Toolchain
- [ ] JDK 21 (Eclipse Temurin)
- [ ] Gradle 9.7.1
- [ ] Fabric Loom 1.17.21 (for MC 1.21.1)
- [ ] Yarn mappings 1.21.1+build.3
- [ ] .NET 8 SDK (for IL2CPP/Cpp2IL extraction)
- [ ] Python 3.11+ with `pip install pillow numpy pyyaml unitypack`

### 0.2 Universal Modder Audit
- [ ] Clone `rehan-remade/universal-modder`
- [ ] Security scan: no telemetry, no auto-update, no obfuscation
- [ ] Verify only `um scan`, `um backup`, `um publish-check` are safe
- [ ] Block: `um win`, `um fal`, `um kb sync/pr`

### 0.3 Backup User Files
- [ ] `%APPDATA%\.minecraft\mods` → `backups/<timestamp>/mods`
- [ ] `%APPDATA%\.minecraft\shaderpacks` → `backups/<timestamp>/shaderpacks`
- [ ] `%APPDATA%\.minecraft\resourcepacks` → `backups/<timestamp>/resourcepacks`
- [ ] `%APPDATA%\.minecraft\config` → `backups/<timestamp>/config`

### 0.4 Modpack Matrix Recon
- [ ] Enumerate all mods in Lunar Client's active profile
- [ ] Check for Fabric API version conflicts
- [ ] Document worldgen mods (Terralith, Biomes O' Plenty, Regions Unexplored, Nature's Spirit)
- [ ] Verify Iris + Sodium compatibility requirements

---

## Phase 1: Data Extraction (M1)

### 1.1 Fishing Planet Asset Extraction
- [ ] Use Cpp2IL to dump Assembly-CSharp.dll from `FishingPlanet_Data/Managed`
- [ ] Parse 5,675+ script objects for:
  - Fish species definitions (name, scientific name, weight range, habitat, behavior)
  - Rod definitions (power, action, length, material, price)
  - Reel definitions (gear ratio, bearings, drag, line capacity)
  - Lure/bait definitions (type, size, color, action, target species)
  - Line definitions (test strength, material, diameter)
  - Hook definitions (size, type, material)
  - Water body definitions (location, fish populations, depths)
- [ ] Extract AssetBundles from `StreamingAssets/AssetBundles/x32/`:
  - Fish models (FBX/OBJ) + textures
  - Rod/reel models + textures
  - Lure models + textures
  - UI icons
- [ ] Extract FMOD sound banks from `StreamingAssets/Banks/`:
  - Splash, reel, drag, catch sounds
  - Ambient water sounds per biome

### 1.2 Data Normalization
- [ ] Convert fish stats to Minecraft scale (weight → health/exp, rarity → spawn weight)
- [ ] Map Fishing Planet water types → Minecraft biomes (ocean, river, lake, swamp, etc.)
- [ ] Create JSON manifest: `assets/fishingplanet/data/fish_manifest.json`

---

## Phase 2: Architecture Design (M2)

### 2.1 Mod Architecture
```
fishingplanet-fabric/
├── src/main/java/com/fishingplanet/
│   ├── FishingPlanetMod.java           # Main entry point
│   ├── registry/
│   │   ├── ModItems.java               # Item registration (generated)
│   │   ├── ModEntities.java            # Fish entity registration
│   │   ├── ModBlocks.java              # Bobber, tackle boxes
│   │   └── ModSounds.java              # Sound events
│   ├── entity/
│   │   ├── fish/
│   │   │   ├── FishingPlanetFishEntity.java
│   │   │   ├── FishSpecies.java        # Data-driven species
│   │   │   └── FishBehavior.java       # AI: schooling, feeding, fleeing
│   │   └── bobber/
│   │       └── FishingPlanetBobberEntity.java
│   ├── item/
│   │   ├── RodItem.java                # Custom rod with tackle slots
│   │   ├── ReelItem.java
│   │   ├── LureItem.java
│   │   ├── LineItem.java
│   │   ├── HookItem.java
│   │   └── TackleBoxItem.java
│   ├── fishing/
│   │   ├── FishingMechanics.java       # Override vanilla fishing
│   │   ├── LootTables.java             # Dynamic loot per species/biome
│   │   └── TackleSystem.java           # Rod+Reel+Line+Lure+Hook combo
│   ├── spawn/
│   │   ├── FishSpawner.java            # Custom spawner for all water
│   │   ├── SpawnConditions.java        # Biome, depth, time, weather
│   │   └── PopulationManager.java      # Density control per chunk
│   ├── config/
│   │   └── ModConfig.java              # Cloth Config API
│   ├── command/
│   │   ├── GiveFishCommand.java
│   │   ├── SpawnFishCommand.java
│   │   └── SetFrequencyCommand.java
│   └── compat/
│       ├── IrisCompat.java             # Shader uniform bindings
│       ├── SodiumCompat.java           # Render layer hooks
│       └── OptimumRealismCompat.java   # PBR texture variants
├── src/main/resources/
│   ├── assets/fishingplanet/
│   │   ├── models/item/                # Generated from extracted assets
│   │   ├── models/entity/              # Fish entity models
│   │   ├── textures/entity/fish/       # LabPBR converted textures
│   │   ├── textures/item/              # Rod, reel, lure icons
│   │   ├── lang/en_us.json
│   │   └── sounds.json
│   ├── data/fishingplanet/
│   │   ├── loot_tables/                # Per-species loot
│   │   ├── fish_spawns/                # Biome-specific spawn configs
│   │   └── tags/                       # Item tags for compatibility
│   └── fabric.mod.json
└── build.gradle.kts
```

### 2.2 Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Data-driven fish species (JSON) | 100+ species, no Java class per fish |
| Custom bobber entity | Fishing Planet tackle physics ≠ vanilla |
| LabPBR textures for Bliss | Optimum Realism uses labPBR; converts to Chocapic/Bliss uniforms |
| Chunk-based population manager | Performance: ~500 fish/chunk max, culls distant |
| Fabric API only | Lunar uses Fabric; no Forge/Mixin conflicts |
| Cloth Config | In-game config GUI, no config file editing |

---

## Phase 3: Minimal Vertical Slice (M3)

### 3.1 Target: 3 Fish Species + 2 Rods Working
- [ ] Species: Largemouth Bass, Rainbow Trout, Atlantic Salmon
- [ ] Rods: Basic Spinning Rod, Basic Casting Rod
- [ ] Basic tackle: 1 line, 1 hook, 1 lure per rod
- [ ] Spawning in: River, Ocean, Swamp biomes
- [ ] Test under: Iris + Bliss + Sodium + Optimum Realism

### 3.2 Acceptance Criteria
- [ ] `/give @p fishingplanet:largemouth_bass` works
- [ ] Fish spawn naturally in water (visible, swimmable)
- [ ] Custom rod catches fish with custom loot table
- [ ] No crashes with Bliss shader enabled
- [ ] Textures render correctly with Optimum Realism
- [ ] FPS impact < 5% with 50 fish loaded

---

## Phase 4: Full Conversion (M4)

### 4.1 Automated Asset Pipeline
- [ ] Python script: `extract_assets.py`
  - Input: Fishing Planet AssetBundles + Cpp2IL dump
  - Output: Minecraft-ready models (JSON), textures (PNG + labPBR), sounds (OGG)
- [ ] Python script: `generate_registry.py`
  - Input: Normalized JSON manifest
  - Output: `ModItems.java`, `ModEntities.java`, loot tables, spawn configs, lang files

### 4.2 Complete Content
- [ ] All ~150 fish species from Fishing Planet
- [ ] All ~50 rods, ~30 reels, ~200 lures, ~20 lines, ~15 hooks
- [ ] All water body mappings → Minecraft biomes + modded biomes (BOP, Terralith, etc.)
- [ ] Sound events for all actions

---

## Phase 5: Gameplay Depth & Polish (M5)

### 5.1 Fishing Mechanics
- [ ] Tackle durability & breakage
- [ ] Fish fighting minigame (tension meter, drag control)
- [ ] Species-specific behavior (bottom feeders, surface, schooling)
- [ ] Time-of-day / weather / season modifiers

### 5.2 Shader & Resource Pack Integration
- [ ] Bliss: Custom fish shader (translucency, scales, caustics)
- [ ] Optimum Realism: PBR texture variants (roughness, metalness, normal)
- [ ] Iris: Custom uniform bindings for fish highlight/shimmer

### 5.3 Commands & Config
- [ ] `/fp give <item> [count]` - give any mod item
- [ ] `/fp spawn <species> [count] [radius]` - spawn fish at player
- [ ] `/fp frequency <multiplier>` - global spawn rate (0.1x - 10x)
- [ ] `/fp density <fish_per_chunk>` - max density cap
- [ ] Config GUI: Cloth Config screen with all options

---

## Phase 6: Adversarial Testing (M6)

### 6.1 Compatibility Matrix
| Test | Mod/Shader | Expected |
|------|------------|----------|
| Boot | Vanilla 1.21.1 + Fabric API | ✅ |
| Boot | Lunar Client profile (55 mods) | ✅ |
| Boot | + Bliss Shader | ✅ |
| Boot | + Optimum Realism | ✅ |
| Boot | + Iris + Sodium + Lithium | ✅ |
| Boot | + Terralith + BOP + Regions Unexplored | ✅ |
| Fish spawn | All vanilla biomes | ✅ |
| Fish spawn | Modded biomes (BOP, Terralith) | ✅ |
| Fishing | Vanilla rod + custom bobber | ✅ |
| Fishing | Custom rod + full tackle | ✅ |
| Render | 500 fish on screen | < 5% FPS drop |
| Memory | 1hr gameplay | No leaks |

### 6.2 Stress Tests
- [ ] 1000 fish spawned via command
- [ ] Chunk loading/unloading with fish
- [ ] Dimension travel (Nether, End, modded dimensions)
- [ ] Multiplayer sync (if applicable)

---

## Phase 7: Install & Handoff (M7)

### 7.1 Final Build
- [ ] `./gradlew build` → `fishingplanet-fabric-1.0.0.jar`
- [ ] Verify jar signature (none - unsigned for Lunar)

### 7.2 Installation
- [ ] Copy jar to `%APPDATA%\.minecraft\mods\`
- [ ] Copy Bliss shader addon to `%APPDATA%\.minecraft\shaderpacks\Bliss_FishingPlanet_Addon.zip`
- [ ] Copy Optimum Realism PBR addon to `%APPDATA%\.minecraft\resourcepacks\OptimumRealism_FishingPlanet_Addon.zip`

### 7.3 Verification
- [ ] Launch via Lunar Client
- [ ] Verify mod appears in Mod Menu
- [ ] Test `/fp give` command
- [ ] Verify fish spawn in ocean/river
- [ ] Verify shader + resource pack visuals

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fishing Planet assets server-side only | High | High | Fallback: wiki/manual data entry for stats; procedural models |
| IL2CPP extraction fails | Medium | High | Use AssetStudio/UnityPack as backup; manual modeling for key species |
| Bliss shader breaks custom entities | Medium | Medium | Test early (M3); fallback to vanilla rendering path |
| Lunar Client mod conflicts | Medium | High | Test against full mod list in M0; use Fabric API only |
| Performance with 100+ fish species | Medium | Medium | LOD system, chunk culling, entity pooling |
| Optimum Realism PBR mismatch | Low | Medium | Provide both labPBR and legacy texture variants |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| M0: Toolchain & Safety | 1 day | None |
| M1: Data Extraction | 2-3 days | M0 |
| M2: Architecture Design | 1 day | M1 |
| M3: Vertical Slice | 2-3 days | M2 |
| M4: Full Conversion | 3-5 days | M3 |
| M5: Polish & Depth | 2-3 days | M4 |
| M6: Adversarial Testing | 2 days | M5 |
| M7: Install & Handoff | 0.5 days | M6 |
| **Total** | **~14-18 days** | |

---

## Next Steps

1. **Confirm plan** - Review and approve/adjust
2. **Start M0** - Install toolchain, audit Universal Modder, backup files
3. **Parallel M1** - Begin Fishing Planet extraction while toolchain installs

---

*Generated: 2026-10-07*
*Status: Awaiting approval to proceed with M0*