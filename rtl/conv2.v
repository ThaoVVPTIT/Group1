`timescale 1ns / 1ps
// =============================================================
// CONV2 : Tich chap 3x3, 6 kenh vao -> 16 kenh ra
// Vao : 6  x 13 x 13  (1014 gia tri, dia chi 10 bit)
// Ra  : 16 x 11 x 11  (1936 gia tri, dia chi 11 bit)
// =============================================================
module conv2 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    input  wire        in_wr_en,
    input  wire [9:0]  in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    input  wire [10:0] out_rd_addr,
    output wire signed [7:0]  out_rd_data
);

    reg signed [7:0]  c2_kernel [0:863];
    reg signed [31:0] c2_bias   [0:15];
    reg signed [31:0] c2_mult   [0:15];
    reg [7:0]         c2_shift  [0:15];

    initial begin
        $readmemh("weights_hex/conv2_kernel.hex",     c2_kernel);
        $readmemh("weights_hex/conv2_bias.hex",       c2_bias);
        $readmemh("weights_hex/conv2_multiplier.hex", c2_mult);
        $readmemh("weights_hex/conv2_shift.hex",      c2_shift);
    end

    reg signed [7:0] p1_out [0:1013];
    reg signed [7:0] c2_out [0:1935];

    always @(posedge clk) begin
        if (in_wr_en) p1_out[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = c2_out[out_rd_addr];

    localparam IDLE = 0, INIT = 1, MAC = 2, QUANT = 3;
    reg [2:0] state;

    reg [15:0] c, y, x, ky, kx, kc;
    reg signed [31:0] acc;
    reg signed [63:0] prod;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        c <= 0; y <= 0; x <= 0;
                        state <= INIT;
                    end
                end

                INIT: begin
                    acc <= c2_bias[c];
                    kc <= 0; ky <= 0; kx <= 0;
                    state <= MAC;
                end

                MAC: begin
                    acc <= acc + p1_out[kc*169 + (y+ky)*13 + (x+kx)] * c2_kernel[c*54 + kc*9 + ky*3 + kx];

                    if (kx == 2) begin
                        kx <= 0;
                        if (ky == 2) begin
                            ky <= 0;
                            if (kc == 5) state <= QUANT;
                            else kc <= kc + 1;
                        end else ky <= ky + 1;
                    end else kx <= kx + 1;
                end

                QUANT: begin
                    prod = acc * c2_mult[c];
                    if (c2_shift[c] > 0) begin
                        prod = prod + (64'sd1 << (c2_shift[c] - 1));
                        acc = prod >>> c2_shift[c];
                    end else acc = prod;

                    if (acc > 127) acc = 127; else if (acc < -128) acc = -128;
                    if (acc < 0) acc = 0;

                    c2_out[c*121 + y*11 + x] <= acc[7:0];

                    if (x == 10) begin
                        x <= 0;
                        if (y == 10) begin
                            y <= 0;
                            if (c == 15) begin
                                done  <= 1;
                                state <= IDLE;
                            end else begin
                                c <= c + 1;
                                state <= INIT;
                            end
                        end else begin y <= y + 1; state <= INIT; end
                    end else begin x <= x + 1; state <= INIT; end
                end
            endcase
        end
    end
endmodule