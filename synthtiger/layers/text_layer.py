"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synthtiger import utils
from synthtiger.layers.layer import Layer
import random

class TextLayer(Layer):
    def __init__(
        self,
        text,
        path,
        size,
        color=(0, 0, 0, 255),
        bold=False,
        vertical=False,
        supersub=False,
    ):
        # https://en.wikipedia.org/wiki/Backslash
        text = text.replace("\\", "＼")

        font = self._read_font(path, size)
        image, bbox = self._render_text(text, font, color, bold, vertical, supersub)

        super().__init__(image)
        self.bbox = bbox

    def _read_font(self, path, size):
        font = ImageFont.truetype(path, size=size)
        return font

    def _render_text(self, text, font, color, bold, vertical, supersub):
        if not vertical:
            if not supersub:
                image, bbox = self._render_hori_text(text, font, color, bold)
            else:
                image, bbox = self._render_supersub_text(text, font, color, bold)
        else:
            image, bbox = self._render_vert_text(text, font, color, bold)

        return image, bbox

    def _render_hori_text(self, text, font, color, bold):
        image, bbox = self._get_image(text, font, color, bold, False)
        return image, bbox

    def _render_vert_text(self, text, font, color, bold):
        chars = utils.split_text(text, reorder=True)
        patches = []
        bboxes = []

        for char in chars:
            patch, bbox = self._render_vert_char(char, font, color, bold)
            patches.append(patch)
            bboxes.append(bbox)

        width = max([bbox[2] for bbox in bboxes])
        height = sum([bbox[3] for bbox in bboxes])
        left = min([bbox[0] for bbox in bboxes])
        bottom = 0

        for bbox in bboxes:
            bbox[0] -= left
            bbox[1] = bottom
            bottom += bbox[3]

        image = utils.create_image((width, height))
        for patch, (x, y, w, h) in zip(patches, bboxes):
            image[y : y + h, x : x + w] = patch

        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_char(self, char, font, color, bold):
        fullwidth_char = utils.to_fullwidth(char)[0]

        if utils.vert_orient(fullwidth_char) != "Tr" and fullwidth_char.isalnum():
            return self._render_vert_upright_char(char, font, color, bold)

        if utils.vert_rot_flip(fullwidth_char):
            return self._render_vert_rot_flip_char(char, font, color, bold)

        if utils.vert_right_flip(fullwidth_char):
            return self._render_vert_right_flip_char(char, font, color, bold)

        if utils.vert_orient(fullwidth_char) in ("R", "Tr"):
            return self._render_vert_rot_char(char, font, color, bold)

        return self._render_vert_upright_char(char, font, color, bold)

    def _render_vert_upright_char(self, char, font, color, bold):
        vertical = len(char) <= 1
        image, bbox = self._get_image(char, font, color, bold, vertical)
        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]
        return image, bbox

    def _render_vert_rot_char(self, char, font, color, bold):
        image, bbox = self._get_image(char, font, color, bold, False)
        image, _ = utils.fit_image(image, left=False, right=False)

        ascent, width = -bbox[1], bbox[2]
        left = max(ascent - width, 0) // 2
        right = max(ascent - width, 0) - left
        image = np.pad(image, ((0, 0), (left, right), (0, 0)))
        image = np.rot90(image, k=-1)

        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_rot_flip_char(self, char, font, color, bold):
        image, bbox = self._get_image(char, font, color, bold, False)

        ascent, width = -bbox[1], bbox[2]
        left = max(ascent - width, 0) // 2
        right = max(ascent - width, 0) - left
        image = np.pad(image, ((0, 0), (left, right), (0, 0)))
        image = np.rot90(image, k=-1)
        image = np.fliplr(image)

        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_right_flip_char(self, char, font, color, bold):
        bbox = self._get_bbox(char, font, False)
        inner_bbox = self._get_inner_bbox(char, font, bold, False)
        sx, sy, patch_width, patch_height = inner_bbox

        patch, _ = self._get_image(char, font, color, bold, False)
        patch = patch[sy : sy + patch_height, sx : sx + patch_width]
        patch_height, patch_width = patch.shape[:2]

        ascent = -bbox[1]
        width, height = max(ascent, patch_width), max(ascent, patch_height)
        dx, dy = max(width - patch_width, 0), max(height - patch_height - sy, 0)

        image = utils.create_image((width, height))
        image[dy : dy + patch_height, dx : dx + patch_width] = patch
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _get_image(self, text, font, color, bold, vertical):
        stroke_width = self._get_stroke_width(bold)
        direction = self._get_direction(vertical)
        bbox = self._get_bbox(text, font, vertical)
        width, height = bbox[2:]

        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, 0),
            text,
            fill=color,
            font=font,
            stroke_width=stroke_width,
            direction=direction,
        )
        image = np.array(image, dtype=np.float32)

        return image, bbox

    def _get_bbox(self, text, font, vertical):
        direction = self._get_direction(vertical)

        if not vertical:
            ascent, descent = font.getmetrics()
            width = font.getsize(text, direction=direction)[0]
            height = ascent + descent
            bbox = [0, -ascent, width, height]
        else:
            width, height = font.getsize(text, direction=direction)
            bbox = [-width // 2, 0, width, height]

        return bbox

    def _get_inner_bbox(self, text, font, bold, vertical):
        stroke_width = self._get_stroke_width(bold)
        direction = self._get_direction(vertical)

        mask, offset = font.getmask2(
            text, stroke_width=stroke_width, direction=direction
        )
        bbox = mask.getbbox()
        left = max(bbox[0] + offset[0], 0)
        top = max(bbox[1] + offset[1], 0)
        right = max(bbox[2] + offset[0], 0)
        bottom = max(bbox[3] + offset[1], 0)
        width = max(right - left, 0)
        height = max(bottom - top, 0)
        bbox = [left, top, width, height]

        return bbox

    def _get_stroke_width(self, bold):
        stroke_width = int(bold)
        return stroke_width

    def _get_direction(self, vertical):
        direction = "ltr" if not vertical else "ttb"
        return direction

    def _render_supersub_text(self, text, font, color, bold):
        CHAR_SPACING_RANGE = (-2, 3)
        SUPERSCRIPT_PROB = 0.15
        SUBSCRIPT_PROB = 0.15
        SUP_SUB_SCALE_RANGE = (0.55, 0.75)
        SUPERSCRIPT_OFFSET_RANGE = (0.3, 0.5)
        SUBSCRIPT_OFFSET_RANGE = (0.3, 0.5)
        
        ascent, descent = font.getmetrics()
        font_height = ascent + descent
        
        chars = utils.split_text(text)
        
        if not chars:
            empty_image = utils.create_image((1, font_height))
            return empty_image, [0, -ascent, 1, font_height]

        char_images = []
        char_positions = []
        
        current_x = 0
        min_y = 0
        max_y = font_height
        max_width = 0
        
        for char in chars:
            rand_val = random.random()
            if rand_val < SUPERSCRIPT_PROB:
                scale = random.uniform(*SUP_SUB_SCALE_RANGE)
                sub_size = max(1, int(font.size * scale))
                sub_font = self._read_font(path=font.path, size=sub_size)
                char_img, char_bbox = self._get_image(char, sub_font, color, bold, False)

                y_offset = -int(font_height * random.uniform(*SUPERSCRIPT_OFFSET_RANGE))
                x_pos = current_x
                y_pos = y_offset
                
                min_y = min(min_y, y_pos)
                max_y = max(max_y, y_pos + char_img.shape[0])
                
            elif rand_val < SUPERSCRIPT_PROB + SUBSCRIPT_PROB:
                scale = random.uniform(*SUP_SUB_SCALE_RANGE)
                sub_size = max(1, int(font.size * scale))
                sub_font = self._read_font(path=font.path, size=sub_size)
                char_img, char_bbox = self._get_image(char, sub_font, color, bold, False)
                
                y_offset = int(font_height * random.uniform(*SUBSCRIPT_OFFSET_RANGE))
                x_pos = current_x
                y_pos = y_offset
                
                min_y = min(min_y, y_pos)
                max_y = max(max_y, y_pos + char_img.shape[0])
                
            else:
                char_img, char_bbox = self._get_image(char, font, color, bold, False)
                x_pos = current_x
                y_pos = 0
                

                max_y = max(max_y, font_height)
            
            char_images.append(char_img)
            char_positions.append((x_pos, y_pos))
            
            char_spacing = random.randint(*CHAR_SPACING_RANGE)
            current_x += char_img.shape[1] + char_spacing
        
        total_width = current_x
        total_height = max_y - min_y

        image = utils.create_image((total_width, total_height))

        for i, (char_img, (x, y)) in enumerate(zip(char_images, char_positions)):
            h, w = char_img.shape[:2]
            adjusted_y = y - min_y
            
            if adjusted_y < 0:
                char_img = char_img[-adjusted_y:]
                h = char_img.shape[0]
                adjusted_y = 0
                
            if adjusted_y + h > total_height:
                char_img = char_img[:total_height - adjusted_y]
                h = char_img.shape[0]
            
            if w > 0 and h > 0:
                dest_w = min(w, total_width - x)
                if dest_w > 0:
                    image[adjusted_y:adjusted_y + h, x:x + dest_w] = char_img[:, :dest_w]
        
        bbox = [0, min_y, total_width, total_height]
        
        return image, bbox