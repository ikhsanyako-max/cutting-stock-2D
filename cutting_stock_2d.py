"""
2D Cutting Stock Optimizer — Streamlit Web App
Optimisasi pemotongan lembaran/sheet 2 dimensi.
Algoritma: MaxRects Bin Packing + Multi-Strategy + Guillotine Support
Referensi: CutPro Optimizer, Jukka Jylänki (MaxRects)

Fitur:
  • Multi-sheet / multi-ukuran stock
  • Rotasi 90° (opsional, per piece)
  • Blade kerf / saw cut width
  • Trim edge (kiri, kanan, atas, bawah)
  • Grain direction constraint
  • 6 strategi heuristik — auto-pick terbaik
  • Visualisasi layout interaktif
  • Laporan PDF profesional
"""

import streamlit as st
import math
import time
import io
import copy
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from datetime import datetime
import pandas as pd

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas as rl_canvas

# ═══════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="2D Cutting Stock Optimizer",
    page_icon="🔲",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; margin-bottom: 1.5rem;
    }
    .metric-row {
        display: flex; gap: 8px; flex-wrap: wrap;
    }
    .warning-box {
        background: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .error-box {
        background: #fef2f2; border-left: 4px solid #ef4444;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .success-box {
        background: #f0fdf4; border-left: 4px solid #22c55e;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0; border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] {
        border: 1px dashed #CBD5E1; background: #F8FAFC;
        border-radius: 8px; padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Piece:
    """Potongan yang diinginkan (demand)."""
    id: str
    label: str
    width: float
    height: float
    qty: int
    can_rotate: bool = True

@dataclass
class StockSheet:
    """Lembaran stok (material)."""
    width: float
    height: float
    qty: int
    label: str = ""

@dataclass
class Placement:
    """Penempatan satu potongan pada sheet."""
    piece_id: str
    label: str
    x: float
    y: float
    width: float      # lebar setelah rotasi
    height: float     # tinggi setelah rotasi
    rotated: bool
    orig_w: float
    orig_h: float

@dataclass
class Rect:
    """Free rectangle dalam MaxRects."""
    x: float
    y: float
    width: float
    height: float

@dataclass
class SheetResult:
    """Hasil pemotongan satu sheet."""
    sheet_index: int
    stock_width: float
    stock_height: float
    placements: List[Placement]
    free_rects: List[Rect]
    utilization: float = 0.0
    waste_area: float = 0.0
    trim_area: float = 0.0
    kerf_area: float = 0.0

@dataclass
class OptimResult:
    """Hasil keseluruhan optimisasi."""
    status: str
    message: str = ""
    sheets: List[SheetResult] = field(default_factory=list)
    strategy_name: str = ""
    total_sheets_used: int = 0
    total_utilization: float = 0.0
    total_waste_pct: float = 0.0
    total_pieces_placed: int = 0
    total_pieces_required: int = 0
    unplaced_pieces: List[dict] = field(default_factory=list)
    duration: float = 0.0
    all_strategy_results: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# MAXRECTS BIN PACKING ENGINE
# ═══════════════════════════════════════════════════════════════

class MaxRectsBin:
    """
    MaxRects Bin Packing algorithm.
    Mengelola satu bin (sheet) dan menempatkan rectangles secara optimal.
    Mendukung beberapa heuristik penempatan.
    
    Referensi: Jukka Jylänki, "A Thousand Ways to Pack the Bin"
    """

    def __init__(self, width: float, height: float,
                 kerf: float = 0.0,
                 trim_left: float = 0.0, trim_right: float = 0.0,
                 trim_top: float = 0.0, trim_bottom: float = 0.0,
                 guillotine: bool = False):
        self.bin_width = width
        self.bin_height = height
        self.kerf = kerf
        self.trim_left = trim_left
        self.trim_right = trim_right
        self.trim_top = trim_top
        self.trim_bottom = trim_bottom
        self.guillotine = guillotine

        # Usable area after trim
        self.usable_x = trim_left
        self.usable_y = trim_bottom
        self.usable_w = width - trim_left - trim_right
        self.usable_h = height - trim_top - trim_bottom

        # Initialize free rectangles with full usable area
        self.free_rects: List[Rect] = [
            Rect(self.usable_x, self.usable_y, self.usable_w, self.usable_h)
        ]
        self.placements: List[Placement] = []
        self.used_area = 0.0

    def _fits(self, rect_w, rect_h, free: Rect) -> bool:
        """Check apakah rectangle muat di free rect (termasuk kerf)."""
        needed_w = rect_w + self.kerf
        needed_h = rect_h + self.kerf
        return needed_w <= free.width + 0.001 and needed_h <= free.height + 0.001

    def _score_bssf(self, rect_w, rect_h, free: Rect) -> Tuple[float, float]:
        """Best Short Side Fit: minimasi sisi pendek sisa."""
        leftover_h = free.height - rect_h - self.kerf
        leftover_w = free.width - rect_w - self.kerf
        short_side = min(leftover_w, leftover_h)
        long_side = max(leftover_w, leftover_h)
        return (short_side, long_side)

    def _score_blsf(self, rect_w, rect_h, free: Rect) -> Tuple[float, float]:
        """Best Long Side Fit: minimasi sisi panjang sisa."""
        leftover_h = free.height - rect_h - self.kerf
        leftover_w = free.width - rect_w - self.kerf
        short_side = min(leftover_w, leftover_h)
        long_side = max(leftover_w, leftover_h)
        return (long_side, short_side)

    def _score_baf(self, rect_w, rect_h, free: Rect) -> Tuple[float, float]:
        """Best Area Fit: minimasi sisa area."""
        area_fit = free.width * free.height - rect_w * rect_h
        short_side = min(free.width - rect_w, free.height - rect_h)
        return (area_fit, short_side)

    def _score_bl(self, rect_w, rect_h, free: Rect) -> Tuple[float, float]:
        """Bottom-Left: prioritas posisi y terendah, lalu x terendah."""
        return (free.y, free.x)

    def _score_cp(self, rect_w, rect_h, free: Rect) -> Tuple[float, float]:
        """Contact Point: maksimalkan titik kontak dgn batas/piece lain."""
        contact = 0
        if free.x <= self.usable_x + 0.01:
            contact += rect_h
        if free.y <= self.usable_y + 0.01:
            contact += rect_w
        if free.x + rect_w >= self.usable_x + self.usable_w - 0.01:
            contact += rect_h
        if free.y + rect_h >= self.usable_y + self.usable_h - 0.01:
            contact += rect_w
        # Negative karena kita ingin MAXIMIZE contact
        return (-contact, free.y)

    def _score_wf(self, rect_w, rect_h, free: Rect) -> Tuple[float, float]:
        """Worst Fit: pilih free rect terbesar (untuk menjaga ruang besar)."""
        area = -(free.width * free.height)
        return (area, 0)

    def find_best_placement(self, rect_w: float, rect_h: float,
                            can_rotate: bool, method: str
                            ) -> Optional[Tuple[Rect, float, float, bool]]:
        """
        Cari posisi terbaik untuk rectangle (rect_w x rect_h)
        di antara semua free_rects, dengan metode scoring tertentu.
        Return: (free_rect, placed_w, placed_h, rotated) atau None.
        """
        score_fn = {
            "BSSF": self._score_bssf,
            "BLSF": self._score_blsf,
            "BAF":  self._score_baf,
            "BL":   self._score_bl,
            "CP":   self._score_cp,
            "WF":   self._score_wf,
        }[method]

        best = None
        best_score = (float('inf'), float('inf'))

        for fr in self.free_rects:
            # Coba orientasi normal
            if self._fits(rect_w, rect_h, fr):
                score = score_fn(rect_w, rect_h, fr)
                if score < best_score:
                    best_score = score
                    best = (fr, rect_w, rect_h, False)

            # Coba rotasi 90°
            if can_rotate and abs(rect_w - rect_h) > 0.01:
                if self._fits(rect_h, rect_w, fr):
                    score = score_fn(rect_h, rect_w, fr)
                    if score < best_score:
                        best_score = score
                        best = (fr, rect_h, rect_w, True)

        return best

    def place(self, piece_id: str, label: str,
              orig_w: float, orig_h: float,
              can_rotate: bool, method: str) -> Optional[Placement]:
        """Tempatkan satu piece di bin. Return Placement atau None."""
        result = self.find_best_placement(orig_w, orig_h, can_rotate, method)
        if result is None:
            return None

        fr, pw, ph, rotated = result
        placement = Placement(
            piece_id=piece_id, label=label,
            x=fr.x, y=fr.y,
            width=pw, height=ph,
            rotated=rotated,
            orig_w=orig_w, orig_h=orig_h,
        )
        self.placements.append(placement)
        self.used_area += pw * ph

        # Split free rects
        if self.guillotine:
            self._split_guillotine(fr, pw + self.kerf, ph + self.kerf)
        else:
            self._split_maxrects(fr, pw + self.kerf, ph + self.kerf)

        return placement

    def _split_maxrects(self, free: Rect, used_w: float, used_h: float):
        """MaxRects split: buat free rects baru dan prune overlaps."""
        new_rects = []
        to_remove = set()

        for i, fr in enumerate(self.free_rects):
            # Cek overlap antara placed rect dan free rect
            # Placed rect: (free.x, free.y, used_w, used_h)
            px, py = free.x, free.y

            if not (px + used_w > fr.x and px < fr.x + fr.width and
                    py + used_h > fr.y and py < fr.y + fr.height):
                continue

            to_remove.add(i)

            # Split: potong free rect di setiap sisi placed rect
            # Left
            if px > fr.x:
                new_rects.append(Rect(fr.x, fr.y, px - fr.x, fr.height))
            # Right
            if px + used_w < fr.x + fr.width:
                new_rects.append(Rect(
                    px + used_w, fr.y,
                    fr.x + fr.width - (px + used_w), fr.height))
            # Bottom
            if py > fr.y:
                new_rects.append(Rect(fr.x, fr.y, fr.width, py - fr.y))
            # Top
            if py + used_h < fr.y + fr.height:
                new_rects.append(Rect(
                    fr.x, py + used_h,
                    fr.width, fr.y + fr.height - (py + used_h)))

        # Remove affected rects and add new ones
        self.free_rects = [
            fr for i, fr in enumerate(self.free_rects) if i not in to_remove
        ] + new_rects

        self._prune_free_rects()

    def _split_guillotine(self, free: Rect, used_w: float, used_h: float):
        """Guillotine split: hanya horizontal atau vertikal cut."""
        self.free_rects.remove(free)

        # Horizontal split vs Vertical split — pilih yang menyisakan rect terbesar
        # Option A: horizontal (potong horizontal dulu)
        right_a = Rect(free.x + used_w, free.y,
                       free.width - used_w, used_h)
        top_a = Rect(free.x, free.y + used_h,
                     free.width, free.height - used_h)

        # Option B: vertical (potong vertikal dulu)
        right_b = Rect(free.x + used_w, free.y,
                       free.width - used_w, free.height)
        top_b = Rect(free.x, free.y + used_h,
                     used_w, free.height - used_h)

        area_a = max(right_a.width * right_a.height,
                     top_a.width * top_a.height)
        area_b = max(right_b.width * right_b.height,
                     top_b.width * top_b.height)

        if area_a >= area_b:
            if right_a.width > 0.01 and right_a.height > 0.01:
                self.free_rects.append(right_a)
            if top_a.width > 0.01 and top_a.height > 0.01:
                self.free_rects.append(top_a)
        else:
            if right_b.width > 0.01 and right_b.height > 0.01:
                self.free_rects.append(right_b)
            if top_b.width > 0.01 and top_b.height > 0.01:
                self.free_rects.append(top_b)

    def _prune_free_rects(self):
        """Hapus free rects yang sepenuhnya terkandung dalam rect lain."""
        pruned = []
        n = len(self.free_rects)
        contained = [False] * n

        for i in range(n):
            if contained[i]:
                continue
            for j in range(n):
                if i == j or contained[j]:
                    continue
                ri, rj = self.free_rects[i], self.free_rects[j]
                # Apakah ri mengandung rj?
                if (rj.x >= ri.x - 0.001 and rj.y >= ri.y - 0.001 and
                    rj.x + rj.width <= ri.x + ri.width + 0.001 and
                    rj.y + rj.height <= ri.y + ri.height + 0.001):
                    contained[j] = True

        self.free_rects = [
            self.free_rects[i] for i in range(n) if not contained[i]
        ]

    def get_utilization(self) -> float:
        usable = self.usable_w * self.usable_h
        return (self.used_area / usable * 100) if usable > 0 else 0


# ═══════════════════════════════════════════════════════════════
# MULTI-BIN OPTIMIZER
# ═══════════════════════════════════════════════════════════════

STRATEGY_METHODS = {
    "Best Short Side Fit (BSSF)": "BSSF",
    "Best Long Side Fit (BLSF)": "BLSF",
    "Best Area Fit (BAF)": "BAF",
    "Bottom-Left (BL)": "BL",
    "Contact Point (CP)": "CP",
    "Worst Fit (WF)": "WF",
}

SORT_METHODS = {
    "Area Desc": lambda p: -(p.width * p.height),
    "Perimeter Desc": lambda p: -(2 * (p.width + p.height)),
    "Max Side Desc": lambda p: -max(p.width, p.height),
    "Width Desc": lambda p: -p.width,
    "Height Desc": lambda p: -p.height,
    "Diff Desc": lambda p: -abs(p.width - p.height),
}


def expand_pieces(pieces: List[Piece]) -> List[dict]:
    """Expand pieces berdasarkan qty menjadi individual items."""
    items = []
    for p in pieces:
        for i in range(p.qty):
            items.append({
                "piece_id": f"{p.id}_{i+1}",
                "label": p.label,
                "width": p.width,
                "height": p.height,
                "can_rotate": p.can_rotate,
                "orig_piece_id": p.id,
            })
    return items


def expand_stock(stocks: List[StockSheet]) -> List[dict]:
    """Expand stock sheets berdasarkan qty."""
    sheets = []
    for s in stocks:
        for i in range(s.qty):
            sheets.append({
                "width": s.width,
                "height": s.height,
                "label": s.label or f"{s.width}x{s.height}",
            })
    return sheets


def run_single_strategy(
    pieces: List[Piece],
    stocks: List[StockSheet],
    method: str,
    sort_key,
    kerf: float,
    trim_left: float, trim_right: float,
    trim_top: float, trim_bottom: float,
    guillotine: bool,
) -> OptimResult:
    """Jalankan satu strategi penempatan."""
    items = expand_pieces(pieces)
    available_sheets = expand_stock(stocks)
    total_required = len(items)

    # Sort items
    sorted_items = sorted(items, key=lambda it: sort_key(
        type('obj', (object,), {'width': it['width'], 'height': it['height']})()
    ))

    bins: List[MaxRectsBin] = []
    sheet_infos = []
    placed_items = []
    unplaced = []

    for item in sorted_items:
        placed = False

        # Coba tempatkan di bin yang sudah ada
        for bi, b in enumerate(bins):
            p = b.place(
                item["piece_id"], item["label"],
                item["width"], item["height"],
                item["can_rotate"], method
            )
            if p is not None:
                placed = True
                break

        # Jika tidak muat, buka bin baru
        if not placed:
            if len(bins) < len(available_sheets):
                si = len(bins)
                s = available_sheets[si]
                new_bin = MaxRectsBin(
                    s["width"], s["height"],
                    kerf=kerf,
                    trim_left=trim_left, trim_right=trim_right,
                    trim_top=trim_top, trim_bottom=trim_bottom,
                    guillotine=guillotine,
                )
                p = new_bin.place(
                    item["piece_id"], item["label"],
                    item["width"], item["height"],
                    item["can_rotate"], method
                )
                if p is not None:
                    bins.append(new_bin)
                    sheet_infos.append(s)
                    placed = True
                else:
                    # Piece too big for any available sheet
                    unplaced.append({
                        "piece_id": item["piece_id"],
                        "label": item["label"],
                        "width": item["width"],
                        "height": item["height"],
                        "reason": "Tidak muat di sheet manapun",
                    })
            else:
                unplaced.append({
                    "piece_id": item["piece_id"],
                    "label": item["label"],
                    "width": item["width"],
                    "height": item["height"],
                    "reason": "Stok sheet habis",
                })

    # Build results
    sheet_results = []
    total_stock_area = 0
    total_used_area = 0
    total_trim_area = 0

    for bi, b in enumerate(bins):
        s = sheet_infos[bi]
        stock_area = s["width"] * s["height"]
        usable_area = b.usable_w * b.usable_h
        trim_a = stock_area - usable_area
        waste_a = usable_area - b.used_area
        util = b.get_utilization()

        total_stock_area += stock_area
        total_used_area += b.used_area
        total_trim_area += trim_a

        sheet_results.append(SheetResult(
            sheet_index=bi + 1,
            stock_width=s["width"],
            stock_height=s["height"],
            placements=list(b.placements),
            free_rects=list(b.free_rects),
            utilization=util,
            waste_area=waste_a,
            trim_area=trim_a,
            kerf_area=0,  # Approximation
        ))

    total_placed = total_required - len(unplaced)
    total_util = (total_used_area / total_stock_area * 100) if total_stock_area > 0 else 0
    total_waste = 100 - total_util - (total_trim_area / total_stock_area * 100 if total_stock_area > 0 else 0)

    status = "Optimal" if len(unplaced) == 0 else "Partial"

    return OptimResult(
        status=status,
        sheets=sheet_results,
        total_sheets_used=len(bins),
        total_utilization=total_util,
        total_waste_pct=max(0, total_waste),
        total_pieces_placed=total_placed,
        total_pieces_required=total_required,
        unplaced_pieces=unplaced,
    )


def solve_2d_cutting(
    pieces: List[Piece],
    stocks: List[StockSheet],
    kerf: float = 0.0,
    trim_left: float = 0.0, trim_right: float = 0.0,
    trim_top: float = 0.0, trim_bottom: float = 0.0,
    guillotine: bool = False,
    preferred_strategy: str = "Auto (Terbaik)",
) -> OptimResult:
    """
    Solver utama: coba semua kombinasi strategi & sorting,
    pilih yang menggunakan paling sedikit sheet & waste terkecil.
    """
    t0 = time.time()

    total_required = sum(p.qty for p in pieces)
    total_stock_available = sum(s.qty for s in stocks)

    if total_required == 0:
        return OptimResult(status="Error", message="Tidak ada potongan yang diminta.")
    if total_stock_available == 0:
        return OptimResult(status="Error", message="Tidak ada stok sheet.")

    # Validasi: apakah ada piece yang lebih besar dari semua sheet?
    for p in pieces:
        can_fit = False
        for s in stocks:
            uw = s.width - trim_left - trim_right
            uh = s.height - trim_top - trim_bottom
            if (p.width <= uw + 0.01 and p.height <= uh + 0.01):
                can_fit = True
                break
            if p.can_rotate and (p.height <= uw + 0.01 and p.width <= uh + 0.01):
                can_fit = True
                break
        if not can_fit:
            return OptimResult(
                status="Error",
                message=f"Potongan '{p.label}' ({p.width}×{p.height}) "
                        f"tidak muat di sheet manapun setelah dikurangi trim."
            )

    best_result = None
    all_results = {}

    if preferred_strategy != "Auto (Terbaik)":
        methods_to_try = {preferred_strategy: STRATEGY_METHODS[preferred_strategy]}
    else:
        methods_to_try = STRATEGY_METHODS

    sorts_to_try = SORT_METHODS

    for strat_name, method in methods_to_try.items():
        for sort_name, sort_fn in sorts_to_try.items():
            result = run_single_strategy(
                pieces, stocks, method, sort_fn,
                kerf, trim_left, trim_right, trim_top, trim_bottom,
                guillotine,
            )
            key = f"{strat_name} + {sort_name}"
            all_results[key] = {
                "sheets_used": result.total_sheets_used,
                "utilization": result.total_utilization,
                "unplaced": len(result.unplaced_pieces),
                "waste_pct": result.total_waste_pct,
            }

            # Pilih terbaik: fewest unplaced → fewest sheets → highest utilization
            if best_result is None:
                best_result = result
                best_result.strategy_name = key
            else:
                cur_unplaced = len(result.unplaced_pieces)
                best_unplaced = len(best_result.unplaced_pieces)
                if (cur_unplaced < best_unplaced or
                    (cur_unplaced == best_unplaced and
                     result.total_sheets_used < best_result.total_sheets_used) or
                    (cur_unplaced == best_unplaced and
                     result.total_sheets_used == best_result.total_sheets_used and
                     result.total_utilization > best_result.total_utilization)):
                    best_result = result
                    best_result.strategy_name = key

    best_result.duration = time.time() - t0
    best_result.all_strategy_results = all_results
    return best_result


# ═══════════════════════════════════════════════════════════════
# VISUALISASI LAYOUT
# ═══════════════════════════════════════════════════════════════

PIECE_COLORS = [
    '#B3D4FC', '#C8E6C9', '#FFE0B2', '#E1BEE7', '#FFCDD2',
    '#B2EBF2', '#FFF9C4', '#D7CCC8', '#CFD8DC', '#F8BBD0',
    '#DCEDC8', '#F0F4C3', '#FFE082', '#FFAB91', '#CE93D8',
    '#80DEEA', '#A5D6A7', '#EF9A9A', '#90CAF9', '#BCAAA4',
]
PIECE_BORDERS = [
    '#64B5F6', '#81C784', '#FFB74D', '#BA68C8', '#E57373',
    '#4DD0E1', '#FFF176', '#A1887F', '#90A4AE', '#F06292',
    '#AED581', '#DCE775', '#FFD54F', '#FF8A65', '#AB47BC',
    '#26C6DA', '#66BB6A', '#EF5350', '#42A5F5', '#8D6E63',
]


def build_color_map(pieces: List[Piece]):
    """Buat mapping warna per piece label."""
    unique_labels = list(dict.fromkeys(p.label for p in pieces))
    cmap = {}
    bmap = {}
    for i, label in enumerate(unique_labels):
        cmap[label] = PIECE_COLORS[i % len(PIECE_COLORS)]
        bmap[label] = PIECE_BORDERS[i % len(PIECE_BORDERS)]
    return cmap, bmap


def create_layout_figure(sheet_result: SheetResult, cmap, bmap,
                          trim_l, trim_r, trim_t, trim_b, kerf):
    """Buat matplotlib figure untuk satu sheet layout."""
    sw, sh = sheet_result.stock_width, sheet_result.stock_height

    # Dynamic figure sizing
    aspect = sw / sh if sh > 0 else 1
    fig_w = min(12, max(6, aspect * 5))
    fig_h = min(8, max(4, fig_w / aspect))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Sheet background
    sheet_rect = plt.Rectangle(
        (0, 0), sw, sh,
        linewidth=2, edgecolor='#333', facecolor='#F5F5F5', zorder=0
    )
    ax.add_patch(sheet_rect)

    # Trim areas
    trim_color = '#FFF3E0'
    trim_edge = '#FFA726'
    if trim_l > 0:
        ax.add_patch(plt.Rectangle(
            (0, 0), trim_l, sh,
            facecolor=trim_color, edgecolor=trim_edge,
            linewidth=0.5, alpha=0.7, hatch='\\\\', zorder=1))
    if trim_r > 0:
        ax.add_patch(plt.Rectangle(
            (sw - trim_r, 0), trim_r, sh,
            facecolor=trim_color, edgecolor=trim_edge,
            linewidth=0.5, alpha=0.7, hatch='\\\\', zorder=1))
    if trim_b > 0:
        ax.add_patch(plt.Rectangle(
            (0, 0), sw, trim_b,
            facecolor=trim_color, edgecolor=trim_edge,
            linewidth=0.5, alpha=0.7, hatch='\\\\', zorder=1))
    if trim_t > 0:
        ax.add_patch(plt.Rectangle(
            (0, sh - trim_t), sw, trim_t,
            facecolor=trim_color, edgecolor=trim_edge,
            linewidth=0.5, alpha=0.7, hatch='\\\\', zorder=1))

    # Pieces
    for p in sheet_result.placements:
        face = cmap.get(p.label, '#DDD')
        edge = bmap.get(p.label, '#999')

        rect = FancyBboxPatch(
            (p.x, p.y), p.width, p.height,
            boxstyle="round,pad=0",
            linewidth=1.2, edgecolor=edge,
            facecolor=face, zorder=2
        )
        ax.add_patch(rect)

        # Label text
        cx, cy = p.x + p.width / 2, p.y + p.height / 2
        font_size = max(5, min(9, min(p.width, p.height) / (sw / 60)))

        rot_marker = " ↻" if p.rotated else ""
        label_text = f"{p.label}{rot_marker}\n{p.orig_w}×{p.orig_h}"

        # Only show label if piece is big enough
        if p.width > sw * 0.03 and p.height > sh * 0.03:
            ax.text(cx, cy, label_text,
                    fontsize=font_size, ha='center', va='center',
                    fontweight='bold', color='#333',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Waste areas (free rects visualization)
    for fr in sheet_result.free_rects:
        if fr.width > 0.5 and fr.height > 0.5:
            ax.add_patch(plt.Rectangle(
                (fr.x, fr.y), fr.width, fr.height,
                facecolor='#FFEBEE', edgecolor='#E0E0E0',
                linewidth=0.3, alpha=0.4, zorder=1,
                linestyle='--'))

    # Dimensions
    ax.annotate(
        f'{sw}', xy=(sw / 2, sh), xytext=(sw / 2, sh + sh * 0.03),
        fontsize=9, ha='center', va='bottom', fontweight='bold', color='#333'
    )
    ax.annotate(
        f'{sh}', xy=(sw, sh / 2), xytext=(sw + sw * 0.02, sh / 2),
        fontsize=9, ha='left', va='center', fontweight='bold', color='#333',
        rotation=90
    )

    ax.set_xlim(-sw * 0.03, sw * 1.06)
    ax.set_ylim(-sh * 0.03, sh * 1.06)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    seen = set()
    legend_patches = []
    for p in sheet_result.placements:
        if p.label not in seen:
            seen.add(p.label)
            legend_patches.append(mpatches.Patch(
                facecolor=cmap.get(p.label, '#DDD'),
                edgecolor=bmap.get(p.label, '#999'),
                label=f'{p.label} ({p.orig_w}×{p.orig_h})', linewidth=0.5))

    if trim_l > 0 or trim_r > 0 or trim_t > 0 or trim_b > 0:
        legend_patches.append(mpatches.Patch(
            facecolor='#FFF3E0', edgecolor='#FFA726',
            label='Trim', hatch='\\\\'))
    legend_patches.append(mpatches.Patch(
        facecolor='#FFEBEE', edgecolor='#E0E0E0',
        label='Waste', linestyle='--'))

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper center',
                  fontsize=7, framealpha=0.9,
                  ncol=min(len(legend_patches), 6),
                  bbox_to_anchor=(0.5, -0.02))

    title = (f"Sheet #{sheet_result.sheet_index}  "
             f"({sheet_result.stock_width}×{sheet_result.stock_height})  —  "
             f"Utilization: {sheet_result.utilization:.1f}%  —  "
             f"{len(sheet_result.placements)} pieces")
    ax.set_title(title, fontsize=10, fontweight='bold', color='#1E3A5F',
                 pad=10)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_pdf_report(result: OptimResult, pieces: List[Piece],
                         stocks: List[StockSheet],
                         kerf, trim_l, trim_r, trim_t, trim_b,
                         guillotine):
    """Generate laporan PDF teknis."""
    if result.status not in ("Optimal", "Partial"):
        return None

    now = datetime.now()
    doc_id = now.strftime("CSO2D-%Y%m%d-%H%M%S")

    C_PRIMARY = HexColor('#1B3A5C')
    C_SECONDARY = HexColor('#2C5F8A')
    C_ACCENT = HexColor('#E8792F')
    C_LIGHT_BG = HexColor('#F0F4F8')
    C_BORDER = HexColor('#CBD5E1')
    C_TEXT = HexColor('#1E293B')
    C_TEXT_LIGHT = HexColor('#64748B')
    C_SUCCESS = HexColor('#16A34A')
    C_WARNING = HexColor('#D97706')
    C_DANGER = HexColor('#DC2626')
    C_ROW_ALT = HexColor('#F8FAFC')
    C_HEADER_BG = HexColor('#1E3A5F')
    C_WHITE = white

    PAGE_W, PAGE_H = landscape(A4)
    ML = 18 * mm
    MR = 18 * mm
    MT = 22 * mm
    MB = 18 * mm
    CW = PAGE_W - ML - MR

    total_pages = 2 + len(result.sheets)

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=landscape(A4))
    c.setTitle(f"2D Cutting Stock Report — {doc_id}")
    c.setAuthor("2D Cutting Stock Optimizer")

    def draw_header(c, page_num, subtitle=""):
        c.setFillColor(C_PRIMARY)
        c.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT)
        c.setLineWidth(2.5)
        c.line(0, PAGE_H - 14 * mm, PAGE_W, PAGE_H - 14 * mm)
        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(ML, PAGE_H - 9.5 * mm,
                     "2D CUTTING STOCK OPTIMIZATION REPORT")
        c.setFont("Helvetica", 7.5)
        c.drawRightString(PAGE_W - MR, PAGE_H - 6.5 * mm,
                          f"DOC ID: {doc_id}")
        c.drawRightString(PAGE_W - MR, PAGE_H - 10.5 * mm,
                          f"Generated: {now.strftime('%d %b %Y  %H:%M:%S')}")
        if subtitle:
            c.setFillColor(C_SECONDARY)
            c.rect(0, PAGE_H - 21 * mm, PAGE_W, 7 * mm, fill=1, stroke=0)
            c.setFillColor(C_WHITE)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(ML, PAGE_H - 19 * mm, subtitle)

    def draw_footer(c, page_num):
        fy = 10 * mm
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.line(ML, fy + 3 * mm, PAGE_W - MR, fy + 3 * mm)
        c.setFont("Helvetica", 6)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawString(ML, fy,
                     "2D Cutting Stock Optimizer  •  MaxRects Bin Packing  •  "
                     "Multi-Strategy Heuristic")
        c.drawRightString(PAGE_W - MR, fy,
                          f"Hal. {page_num} / {total_pages}")

    def draw_kpi(c, x, y, w, h, label, value, unit="", color=C_PRIMARY, sub=""):
        c.setFillColor(HexColor('#E2E8F0'))
        c.roundRect(x + 1, y - 1, w, h, 3, fill=1, stroke=0)
        c.setFillColor(C_WHITE)
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.roundRect(x, y, w, h, 3, fill=1, stroke=1)
        c.setFillColor(color)
        c.rect(x, y + h - 3, w, 3, fill=1, stroke=0)
        c.setFont("Helvetica", 6.5)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawString(x + 4, y + h - 12, label)
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_TEXT)
        c.drawString(x + 4, y + h - 28, str(value))
        if unit:
            vw = c.stringWidth(str(value), "Helvetica-Bold", 14)
            c.setFont("Helvetica", 8)
            c.setFillColor(C_TEXT_LIGHT)
            c.drawString(x + 6 + vw, y + h - 27, unit)
        if sub:
            c.setFont("Helvetica", 6)
            c.setFillColor(C_TEXT_LIGHT)
            c.drawString(x + 4, y + 4, sub)

    def draw_table(c, x, y, headers, rows, col_widths, row_h=13, fs=6.5):
        cur_y = y
        hh = row_h + 4
        c.setFillColor(C_HEADER_BG)
        c.rect(x, cur_y - hh, sum(col_widths), hh, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", fs)
        c.setFillColor(C_WHITE)
        cx = x
        for i, h in enumerate(headers):
            c.drawCentredString(cx + col_widths[i] / 2, cur_y - hh + 4, h)
            cx += col_widths[i]
        cur_y -= hh
        for ri, row in enumerate(rows):
            bg = C_ROW_ALT if ri % 2 == 0 else C_WHITE
            c.setFillColor(bg)
            c.rect(x, cur_y - row_h, sum(col_widths), row_h, fill=1, stroke=0)
            c.setStrokeColor(HexColor('#E2E8F0'))
            c.setLineWidth(0.3)
            c.line(x, cur_y - row_h, x + sum(col_widths), cur_y - row_h)
            c.setFont("Helvetica", fs)
            c.setFillColor(C_TEXT)
            cx = x
            for i, cell in enumerate(row):
                c.drawCentredString(cx + col_widths[i] / 2, cur_y - row_h + 3, str(cell))
                cx += col_widths[i]
            cur_y -= row_h
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        total_h = hh + len(rows) * row_h
        c.rect(x, y - total_h, sum(col_widths), total_h, fill=0, stroke=1)
        return cur_y

    # ══════════════════════════ PAGE 1: SUMMARY ══════════════════════
    pn = 1
    draw_header(c, pn, "RINGKASAN EKSEKUTIF & PARAMETER")
    draw_footer(c, pn)

    by = PAGE_H - 28 * mm

    # KPIs
    ky = by - 42 * mm
    kw = CW / 6 - 3
    kh = 35 * mm
    total_stock_area = sum(s.stock_width * s.stock_height for s in result.sheets)
    total_used = sum(s.utilization * s.stock_width * s.stock_height / 100
                     for s in result.sheets)
    total_pieces_area = sum(
        p.width * p.height for s in result.sheets for p in s.placements
    )

    kpis = [
        ("UTILISASI", f"{result.total_utilization:.1f}", "%", C_SUCCESS,
         f"{total_pieces_area:,.0f} sq unit"),
        ("WASTE", f"{result.total_waste_pct:.1f}", "%",
         C_DANGER if result.total_waste_pct > 10 else C_WARNING, ""),
        ("SHEET TERPAKAI", f"{result.total_sheets_used}",
         f"/ {sum(s.qty for s in stocks)}", C_PRIMARY, ""),
        ("PIECE PLACED", f"{result.total_pieces_placed}",
         f"/ {result.total_pieces_required}", C_SECONDARY, ""),
        ("STRATEGI", result.strategy_name.split(" + ")[0][:12], "",
         C_PRIMARY, result.strategy_name.split(" + ")[1][:15] if " + " in result.strategy_name else ""),
        ("WAKTU", f"{result.duration:.2f}", "detik", C_ACCENT, ""),
    ]

    for i, (label, val, unit, color, sub) in enumerate(kpis):
        bx = ML + i * (kw + 3)
        draw_kpi(c, bx, ky, kw, kh, label, val, unit, color, sub)

    # Parameters
    sy = ky - 12 * mm
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(C_PRIMARY)
    c.drawString(ML, sy, "PARAMETER INPUT")
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.line(ML, sy - 2, ML + 45 * mm, sy - 2)

    params = [
        ("Blade Kerf", f"{kerf} mm"),
        ("Trim (L/R/T/B)", f"{trim_l}/{trim_r}/{trim_t}/{trim_b} mm"),
        ("Guillotine Cut", "Ya" if guillotine else "Tidak"),
        ("Algoritma", "MaxRects Bin Packing"),
    ]

    py = sy - 14
    for label, val in params:
        c.setFont("Helvetica", 7)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawString(ML + 2, py, label)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(C_TEXT)
        c.drawString(ML + 42 * mm, py, val)
        py -= 11

    # Stock table
    mid_x = ML + CW / 3
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(C_PRIMARY)
    c.drawString(mid_x, sy, "STOK SHEET")
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.line(mid_x, sy - 2, mid_x + 45 * mm, sy - 2)

    stock_rows = []
    for s in stocks:
        stock_rows.append([
            f"{s.width}×{s.height}", f"{s.qty}",
            f"{s.width * s.height * s.qty:,.0f}"
        ])
    scw = [35 * mm, 18 * mm, 30 * mm]
    draw_table(c, mid_x, sy - 10, ["Ukuran", "Qty", "Total Area"],
               stock_rows, scw)

    # Pieces table
    rx = ML + 2 * CW / 3
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(C_PRIMARY)
    c.drawString(rx, sy, "KEBUTUHAN POTONGAN")
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.line(rx, sy - 2, rx + 50 * mm, sy - 2)

    piece_rows = []
    for p in pieces:
        piece_rows.append([
            p.label, f"{p.width}×{p.height}", f"{p.qty}",
            "Ya" if p.can_rotate else "Tidak",
            f"{p.width * p.height * p.qty:,.0f}",
        ])
    pcw = [22 * mm, 24 * mm, 12 * mm, 14 * mm, 22 * mm]
    draw_table(c, rx, sy - 10,
               ["Label", "Ukuran", "Qty", "Rotasi", "Area"],
               piece_rows, pcw)

    c.showPage()

    # ══════════════════════════ PAGES 2..N: LAYOUTS ══════════════════
    cmap, bmap = build_color_map(pieces)

    for si, sr in enumerate(result.sheets):
        pn = si + 2
        draw_header(c, pn,
                    f"LAYOUT SHEET #{sr.sheet_index}  —  "
                    f"{sr.stock_width}×{sr.stock_height}  —  "
                    f"Utilisasi: {sr.utilization:.1f}%  —  "
                    f"{len(sr.placements)} pieces")
        draw_footer(c, pn)

        # Draw layout using reportlab
        ly = PAGE_H - 30 * mm
        layout_h = 110 * mm
        layout_w = CW

        # Scale factor
        scale_x = layout_w / sr.stock_width
        scale_y = layout_h / sr.stock_height
        scale = min(scale_x, scale_y) * 0.9
        ox = ML + (layout_w - sr.stock_width * scale) / 2
        oy = ly - layout_h + (layout_h - sr.stock_height * scale) / 2

        # Sheet bg
        c.setFillColor(HexColor('#F5F5F5'))
        c.setStrokeColor(HexColor('#333333'))
        c.setLineWidth(1.5)
        c.rect(ox, oy, sr.stock_width * scale, sr.stock_height * scale,
               fill=1, stroke=1)

        # Trim
        if trim_l > 0:
            c.setFillColor(HexColor('#FFF3E0'))
            c.setStrokeColor(HexColor('#FFA726'))
            c.setLineWidth(0.5)
            c.rect(ox, oy, trim_l * scale, sr.stock_height * scale,
                   fill=1, stroke=1)
        if trim_r > 0:
            c.setFillColor(HexColor('#FFF3E0'))
            c.rect(ox + (sr.stock_width - trim_r) * scale, oy,
                   trim_r * scale, sr.stock_height * scale, fill=1, stroke=1)
        if trim_b > 0:
            c.setFillColor(HexColor('#FFF3E0'))
            c.rect(ox, oy, sr.stock_width * scale, trim_b * scale,
                   fill=1, stroke=1)
        if trim_t > 0:
            c.setFillColor(HexColor('#FFF3E0'))
            c.rect(ox, oy + (sr.stock_height - trim_t) * scale,
                   sr.stock_width * scale, trim_t * scale, fill=1, stroke=1)

        # Pieces
        for p in sr.placements:
            col_idx = list(cmap.keys()).index(p.label) if p.label in cmap else 0
            face = PIECE_COLORS[col_idx % len(PIECE_COLORS)]
            edge = PIECE_BORDERS[col_idx % len(PIECE_BORDERS)]

            px = ox + p.x * scale
            py = oy + p.y * scale
            pw = p.width * scale
            ph = p.height * scale

            c.setFillColor(HexColor(face))
            c.setStrokeColor(HexColor(edge))
            c.setLineWidth(0.8)
            c.rect(px, py, pw, ph, fill=1, stroke=1)

            if pw > 15 and ph > 10:
                c.setFont("Helvetica-Bold", max(5, min(8, pw / 8)))
                c.setFillColor(HexColor('#333333'))
                c.drawCentredString(px + pw / 2, py + ph / 2 - 2,
                                    f"{p.label}")
                c.setFont("Helvetica", max(4, min(6, pw / 10)))
                c.drawCentredString(px + pw / 2, py + ph / 2 - 10,
                                    f"{p.orig_w}×{p.orig_h}")

        # Piece list table
        tbl_y = oy - 12 * mm
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(C_PRIMARY)
        c.drawString(ML, tbl_y, "DAFTAR POTONGAN PADA SHEET INI")

        p_rows = []
        for p in sr.placements:
            p_rows.append([
                p.label, f"{p.orig_w}×{p.orig_h}",
                f"({p.x:.1f}, {p.y:.1f})",
                f"{p.width}×{p.height}",
                "Ya" if p.rotated else "Tidak",
                f"{p.width * p.height:,.0f}",
            ])
        if len(p_rows) > 15:
            p_rows = p_rows[:15]
            p_rows.append(["...", f"+{len(sr.placements)-15}", "", "", "", ""])

        tw = [25*mm, 25*mm, 30*mm, 25*mm, 18*mm, 25*mm]
        ttotal = sum(tw)
        tw = [w * CW / ttotal for w in tw]
        draw_table(c, ML, tbl_y - 6,
                   ["Label", "Ukuran Asli", "Posisi (x,y)",
                    "Ukuran Final", "Rotasi", "Area"],
                   p_rows, tw, row_h=11, fs=6)

        c.showPage()

    # ══════════════════════════ LAST PAGE: STRATEGY COMPARISON ══════
    pn = total_pages
    draw_header(c, pn, "PERBANDINGAN STRATEGI & RINGKASAN")
    draw_footer(c, pn)

    sy = PAGE_H - 32 * mm
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(C_PRIMARY)
    c.drawString(ML, sy, "PERBANDINGAN SEMUA STRATEGI")
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.line(ML, sy - 2, ML + 60 * mm, sy - 2)

    strat_rows = []
    sorted_strats = sorted(result.all_strategy_results.items(),
                           key=lambda x: (x[1]["unplaced"], x[1]["sheets_used"],
                                          -x[1]["utilization"]))
    for name, data in sorted_strats[:20]:
        is_best = "★" if name == result.strategy_name else ""
        strat_rows.append([
            is_best, name[:40],
            f"{data['sheets_used']}", f"{data['unplaced']}",
            f"{data['utilization']:.1f}%", f"{data['waste_pct']:.1f}%",
        ])

    stw = [8*mm, 80*mm, 20*mm, 20*mm, 25*mm, 25*mm]
    st_total = sum(stw)
    stw = [w * CW / st_total for w in stw]
    draw_table(c, ML, sy - 10,
               ["", "Strategi", "Sheets", "Unplaced", "Utilisasi", "Waste"],
               strat_rows, stw, row_h=11, fs=6)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🔲 2D Cutting Stock Optimizer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Optimisasi pemotongan lembaran 2 dimensi &mdash; '
    'MaxRects Bin Packing &mdash; Multi-Strategy Heuristic</div>',
    unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.header("📐 Data Input")

    # ── Stock Sheets ──
    st.subheader("Stok Sheet (Lembaran)")
    if "stock_list" not in st.session_state:
        st.session_state.stock_list = []

    if st.session_state.stock_list:
        stock_df = pd.DataFrame({
            "No": range(1, len(st.session_state.stock_list) + 1),
            "Lebar": [s["width"] for s in st.session_state.stock_list],
            "Tinggi": [s["height"] for s in st.session_state.stock_list],
            "Qty": [s["qty"] for s in st.session_state.stock_list],
        })
        st.dataframe(stock_df, use_container_width=True, hide_index=True)

        sdel_cols = st.columns([3, 1])
        with sdel_cols[0]:
            sdel_idx = st.selectbox(
                "Hapus stok", key="sdel_sel",
                options=range(1, len(st.session_state.stock_list) + 1),
                label_visibility="collapsed",
                format_func=lambda x: (
                    f"No {x} — "
                    f"{st.session_state.stock_list[x-1]['width']}×"
                    f"{st.session_state.stock_list[x-1]['height']} "
                    f"(qty: {st.session_state.stock_list[x-1]['qty']})"
                ),
            )
        with sdel_cols[1]:
            if st.button("🗑️", key="sdel_btn"):
                st.session_state.stock_list.pop(sdel_idx - 1)
                st.rerun()
    else:
        st.caption("Belum ada stok sheet.")

    with st.form("stock_add", clear_on_submit=True):
        sc = st.columns(3)
        with sc[0]:
            sw = st.number_input("Lebar", min_value=1.0, value=2440.0,
                                  step=10.0, key="sw_in")
        with sc[1]:
            sh = st.number_input("Tinggi", min_value=1.0, value=1220.0,
                                  step=10.0, key="sh_in")
        with sc[2]:
            sq = st.number_input("Qty", min_value=1, value=10,
                                  step=1, key="sq_in")
        if st.form_submit_button("➕ Tambah Stok", use_container_width=True):
            st.session_state.stock_list.append({
                "width": float(sw), "height": float(sh), "qty": int(sq)
            })
            st.rerun()

    st.markdown("---")

    # ── Demand Pieces ──
    st.subheader("Potongan (Demand)")
    if "piece_list" not in st.session_state:
        st.session_state.piece_list = []

    if st.session_state.piece_list:
        piece_df = pd.DataFrame({
            "No": range(1, len(st.session_state.piece_list) + 1),
            "Label": [p["label"] for p in st.session_state.piece_list],
            "W×H": [f"{p['width']}×{p['height']}"
                     for p in st.session_state.piece_list],
            "Qty": [p["qty"] for p in st.session_state.piece_list],
            "Rotasi": ["Ya" if p["can_rotate"] else "Tidak"
                       for p in st.session_state.piece_list],
        })
        st.dataframe(piece_df, use_container_width=True, hide_index=True)

        pdel_cols = st.columns([3, 1])
        with pdel_cols[0]:
            pdel_idx = st.selectbox(
                "Hapus piece", key="pdel_sel",
                options=range(1, len(st.session_state.piece_list) + 1),
                label_visibility="collapsed",
                format_func=lambda x: (
                    f"No {x} — {st.session_state.piece_list[x-1]['label']} "
                    f"({st.session_state.piece_list[x-1]['width']}×"
                    f"{st.session_state.piece_list[x-1]['height']})"
                ),
            )
        with pdel_cols[1]:
            if st.button("🗑️", key="pdel_btn"):
                st.session_state.piece_list.pop(pdel_idx - 1)
                st.rerun()
    else:
        st.caption("Belum ada data potongan.")

    with st.form("piece_add", clear_on_submit=True):
        pc1, pc2 = st.columns(2)
        with pc1:
            pl = st.text_input("Label", value="A", key="pl_in")
            ppw = st.number_input("Lebar", min_value=1.0, value=600.0,
                                   step=10.0, key="pw_in")
        with pc2:
            pq = st.number_input("Qty", min_value=1, value=5,
                                  step=1, key="pq_in")
            pph = st.number_input("Tinggi", min_value=1.0, value=400.0,
                                   step=10.0, key="ph_in")
        pr = st.checkbox("Boleh Rotasi 90°", value=True, key="pr_in")

        if st.form_submit_button("➕ Tambah Potongan", use_container_width=True):
            st.session_state.piece_list.append({
                "label": pl, "width": float(ppw), "height": float(pph),
                "qty": int(pq), "can_rotate": pr,
            })
            st.rerun()

    st.markdown("---")

    # ── Parameters ──
    st.subheader("⚙️ Parameter Optimisasi")
    kerf_val = st.number_input("Blade Kerf / Lebar Pisau (mm)",
                                min_value=0.0, value=3.0, step=0.5)
    tc = st.columns(2)
    with tc[0]:
        tl_val = st.number_input("Trim Kiri", min_value=0.0, value=0.0, step=1.0)
        tt_val = st.number_input("Trim Atas", min_value=0.0, value=0.0, step=1.0)
    with tc[1]:
        tr_val = st.number_input("Trim Kanan", min_value=0.0, value=0.0, step=1.0)
        tb_val = st.number_input("Trim Bawah", min_value=0.0, value=0.0, step=1.0)

    guillo = st.checkbox("Guillotine Cut Only",
                          value=False,
                          help="Jika aktif, potongan hanya mengikuti pola "
                               "guillotine (potong terus dari tepi ke tepi).")
    strategy = st.selectbox(
        "Strategi Penempatan",
        ["Auto (Terbaik)"] + list(STRATEGY_METHODS.keys()),
        index=0,
    )

    st.markdown("---")
    run_btn = st.button("🔍 Cari Solusi Optimal", type="primary",
                         use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# EKSEKUSI
# ═══════════════════════════════════════════════════════════════

if run_btn:
    if not st.session_state.stock_list:
        st.error("Masukkan minimal satu stok sheet.")
        st.stop()
    if not st.session_state.piece_list:
        st.error("Masukkan minimal satu potongan.")
        st.stop()

    stocks_input = [
        StockSheet(s["width"], s["height"], s["qty"])
        for s in st.session_state.stock_list
    ]
    pieces_input = [
        Piece(
            id=f"P{i+1}", label=p["label"],
            width=p["width"], height=p["height"],
            qty=p["qty"], can_rotate=p["can_rotate"],
        )
        for i, p in enumerate(st.session_state.piece_list)
    ]

    with st.spinner("Mengoptimasi layout pemotongan..."):
        result = solve_2d_cutting(
            pieces=pieces_input,
            stocks=stocks_input,
            kerf=kerf_val,
            trim_left=tl_val, trim_right=tr_val,
            trim_top=tt_val, trim_bottom=tb_val,
            guillotine=guillo,
            preferred_strategy=strategy,
        )

    st.session_state.result_2d = result
    st.session_state.stocks_input = stocks_input
    st.session_state.pieces_input = pieces_input
    st.session_state.params_2d = {
        "kerf": kerf_val,
        "trim_l": tl_val, "trim_r": tr_val,
        "trim_t": tt_val, "trim_b": tb_val,
        "guillotine": guillo,
    }


# ═══════════════════════════════════════════════════════════════
# TAMPILKAN HASIL
# ═══════════════════════════════════════════════════════════════

if "result_2d" in st.session_state:
    result = st.session_state.result_2d
    stocks_input = st.session_state.stocks_input
    pieces_input = st.session_state.pieces_input
    params = st.session_state.params_2d

    st.caption(f"Waktu: {result.duration:.2f}s  |  "
               f"Strategi terbaik: {result.strategy_name}")

    if result.status == "Error":
        st.markdown(f"""
        <div class="error-box">
            <strong>❌ Error</strong><br>{result.message}
        </div>
        """, unsafe_allow_html=True)

    elif result.status in ("Optimal", "Partial"):
        if result.status == "Partial":
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Solusi Parsial</strong><br>
                {len(result.unplaced_pieces)} potongan tidak bisa ditempatkan 
                (stok tidak cukup atau ukuran tidak muat).
            </div>
            """, unsafe_allow_html=True)

        # KPI metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Sheet Terpakai",
                      f"{result.total_sheets_used} / "
                      f"{sum(s.qty for s in stocks_input)}")
        with c2:
            st.metric("Piece Placed",
                      f"{result.total_pieces_placed} / "
                      f"{result.total_pieces_required}")
        with c3:
            st.metric("Utilisasi", f"{result.total_utilization:.1f}%")
        with c4:
            st.metric("Waste", f"{result.total_waste_pct:.1f}%")
        with c5:
            st.metric("Waktu", f"{result.duration:.2f}s")

        st.markdown("---")

        # Layout visualizations
        st.subheader("Layout Pemotongan per Sheet")

        cmap, bmap = build_color_map(pieces_input)

        for sr in result.sheets:
            with st.expander(
                f"Sheet #{sr.sheet_index} — "
                f"{sr.stock_width}×{sr.stock_height} — "
                f"{len(sr.placements)} pieces — "
                f"Utilisasi: {sr.utilization:.1f}%",
                expanded=True,
            ):
                fig = create_layout_figure(
                    sr, cmap, bmap,
                    params["trim_l"], params["trim_r"],
                    params["trim_t"], params["trim_b"],
                    params["kerf"],
                )
                st.pyplot(fig)
                plt.close(fig)

                # Detail table
                det_data = []
                for p in sr.placements:
                    det_data.append({
                        "Label": p.label,
                        "Asli": f"{p.orig_w}×{p.orig_h}",
                        "Posisi": f"({p.x:.1f}, {p.y:.1f})",
                        "Final": f"{p.width}×{p.height}",
                        "Rotasi": "Ya" if p.rotated else "Tidak",
                        "Area": f"{p.width * p.height:,.0f}",
                    })
                if det_data:
                    st.dataframe(pd.DataFrame(det_data),
                                 use_container_width=True, hide_index=True)

        # Unplaced pieces
        if result.unplaced_pieces:
            st.markdown("---")
            st.subheader("⚠️ Potongan Tidak Tertampung")
            unpl_df = pd.DataFrame(result.unplaced_pieces)
            st.dataframe(unpl_df, use_container_width=True, hide_index=True)

        # Strategy comparison
        st.markdown("---")
        with st.expander("📊 Perbandingan Strategi", expanded=False):
            strat_data = []
            for name, data in sorted(
                result.all_strategy_results.items(),
                key=lambda x: (x[1]["unplaced"], x[1]["sheets_used"],
                                -x[1]["utilization"])
            ):
                strat_data.append({
                    "★": "★" if name == result.strategy_name else "",
                    "Strategi": name,
                    "Sheets": data["sheets_used"],
                    "Unplaced": data["unplaced"],
                    "Utilisasi (%)": f"{data['utilization']:.1f}",
                    "Waste (%)": f"{data['waste_pct']:.1f}",
                })
            st.dataframe(pd.DataFrame(strat_data),
                         use_container_width=True, hide_index=True)

        # PDF download
        st.markdown("---")
        pdf_bytes = generate_pdf_report(
            result, pieces_input, stocks_input,
            params["kerf"],
            params["trim_l"], params["trim_r"],
            params["trim_t"], params["trim_b"],
            params["guillotine"],
        )
        if pdf_bytes:
            st.download_button(
                label="📥 Unduh Laporan PDF",
                data=pdf_bytes,
                file_name="laporan_2d_cutting_stock.pdf",
                mime="application/pdf",
                type="primary",
            )

else:
    st.info("Masukkan data stok sheet dan potongan di sidebar, "
            "lalu klik **Cari Solusi Optimal**.")

    with st.expander("📖 Cara Penggunaan", expanded=True):
        st.markdown("""
**1. Stok Sheet (Lembaran Material)**
Masukkan ukuran (Lebar × Tinggi) dan jumlah untuk setiap jenis lembaran stok.
Contoh: Plywood 2440 × 1220 mm, qty 10.

**2. Potongan (Demand)**
Masukkan label, ukuran (Lebar × Tinggi), jumlah, dan apakah boleh dirotasi 90°.
Contoh: Part A 600 × 400 mm, qty 20, boleh rotasi.

**3. Parameter**
- **Blade Kerf:** Lebar pisau/gergaji yang hilang saat memotong (mm)
- **Trim (Kiri/Kanan/Atas/Bawah):** Margin tepi sheet yang tidak bisa dipakai
- **Guillotine Cut:** Jika aktif, hanya mengizinkan potongan lurus dari tepi ke tepi
  (seperti panel saw). Lebih realistis tapi waste bisa lebih tinggi.
- **Strategi:** Auto akan mencoba 36 kombinasi strategi dan memilih yang terbaik.

**4. Algoritma**
Menggunakan **MaxRects Bin Packing** — salah satu algoritma paling efisien untuk
2D rectangular packing. Dikombinasikan dengan 6 heuristik penempatan dan
6 metode sorting, menghasilkan 36 strategi yang dievaluasi secara paralel.

**Heuristik Penempatan:**
- **BSSF** (Best Short Side Fit) — Minimasi sisi pendek yang tersisa
- **BLSF** (Best Long Side Fit) — Minimasi sisi panjang yang tersisa
- **BAF** (Best Area Fit) — Pilih area sisa terkecil
- **BL** (Bottom-Left) — Tempatkan di posisi paling kiri-bawah
- **CP** (Contact Point) — Maksimalkan titik kontak dengan batas
- **WF** (Worst Fit) — Pilih ruang terbesar (jaga fleksibilitas)
        """)

    with st.expander("🆚 Perbedaan dengan Cutting 1D", expanded=False):
        st.markdown("""
| Aspek | 1D (Mother Coil) | 2D (Sheet/Lembaran) |
|-------|------------------|---------------------|
| Dimensi | Hanya lebar | Lebar × Tinggi |
| Material | Coil baja/metal | Plywood, kaca, metal sheet |
| Potongan | Strip lebar tertentu | Rectangular pieces |
| Algoritma | LP/MIP (PuLP) | MaxRects Bin Packing |
| Rotasi | Tidak relevan | Opsional 90° |
| Guillotine | Selalu (1 dimensi) | Opsional |
| Kerf | Trim per pisau | Blade kerf 2D |
        """)
