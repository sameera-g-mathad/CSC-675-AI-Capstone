import { StyleSheet, Text, View } from "react-native";
import React, { memo, PropsWithChildren } from "react";

interface chatInterface {
    role: 'user' | 'bot'
}
const colorsToUse = {
    'user': {
        color: '#6366f1',
        borderColor: '#818cf8',
        backgroundColor: '#c7d2fe'
    },
    'bot': {
        color: '#d946ef',
        borderColor: '#e879f9',
        backgroundColor: '#f5d0fe'
    }
}
function Chat({ children, role }: PropsWithChildren & chatInterface) {
    const { color, borderColor, backgroundColor } = colorsToUse[role];
    return (
        <View style={styles.viewContainer}>
            <View style={{ ...styles.msgContainer, borderLeftColor: borderColor, backgroundColor }}>
                <Text style={{ ...styles.roleStyle, color }}>{role}</Text>
                {children}
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        margin: 10
    },
    msgContainer: {
        padding: 10,
        borderLeftWidth: 5,
        backgroundColor: '#e2e8f0',
    },
    roleStyle: {
        fontSize: 17,
        fontWeight: 'bold',
        paddingVertical: 5,
        textTransform: 'capitalize'
    }
});


export default memo(Chat)